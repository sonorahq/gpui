use crate::{Bounds, Half, scene::PrimitiveKind};
use std::{
    cmp,
    fmt::Debug,
    ops::{Add, Sub},
    ptr::NonNull,
};

/// Maximum children per internal node (R-tree style branching factor).
/// Higher values = shorter tree = fewer cache misses, but more work per node.
const MAX_CHILDREN: usize = 12;

/// A spatial tree optimized for finding maximum ordering among intersecting bounds.
///
/// This is an R-tree variant specifically designed for the use case of assigning
/// z-order to overlapping UI elements. Each bounds has an ordering and a kind;
/// bounds sharing an ordering are drawn in the order of their kinds. Key
/// optimizations:
/// - Tracks the leaf with global max key for O(1) fast-path queries
/// - Uses higher branching factor (4) for lower tree height
/// - Aggressive pruning during search based on max_key metadata
#[derive(Debug)]
pub(crate) struct BoundsTree<U>
where
    U: Clone + Debug + Default + PartialEq,
{
    /// All nodes stored contiguously for cache efficiency.
    nodes: Vec<Node<U>>,
    /// Index of the root node, if any.
    root: Option<usize>,
    /// Index of the leaf with the highest key (for fast-path lookups).
    max_leaf: Option<usize>,
    /// An ordering every insert has to stay above, raised while a layer is open.
    floor: u32,
    /// The ordering each open layer starts at and the floor it raised, innermost last.
    open: Vec<(u32, u32)>,
    /// The orderings closed layers hold, inclusive, sorted, with nested ones folded into the
    /// layer around them and back-to-back ones joined. An insert that lands in one is lifted
    /// past it.
    reserved: Vec<(u32, u32)>,
    /// Reusable stack for tree traversal during insertion.
    insert_path: Vec<usize>,
    /// Reusable stack for search operations.
    search_stack: Vec<NonNull<Node<U>>>,
    /// How many orderings were searched for since the tree was made.
    #[cfg(test)]
    pub(crate) searches: usize,
    /// How many bounds went in through [`BoundsTree::insert_known`] since the tree was made.
    #[cfg(test)]
    pub(crate) known_inserts: usize,
}

/// A node in the bounds tree.
#[derive(Debug, Clone)]
struct Node<U>
where
    U: Clone + Debug + Default + PartialEq,
{
    /// Bounding box containing this node and all descendants.
    bounds: Bounds<U>,
    /// Maximum key in this subtree.
    max_key: Key,
    /// Node-specific data.
    kind: NodeKind,
}

/// The ordering of a bounds, then its kind, which orders the bounds that share
/// an ordering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    ordering: u32,
    kind: PrimitiveKind,
}

#[derive(Debug, Clone)]
enum NodeKind {
    /// Leaf node containing actual bounds data.
    Leaf { key: Key },
    /// Internal node with children.
    Internal {
        /// Indices of child nodes (2 to MAX_CHILDREN).
        children: NodeChildren,
    },
}

/// Fixed-size array for child indices, avoiding heap allocation.
#[derive(Debug, Clone)]
struct NodeChildren {
    // Keeps an invariant where the max key child is always at the end
    indices: [usize; MAX_CHILDREN],
    len: u8,
}

impl NodeChildren {
    fn new() -> Self {
        Self {
            indices: [0; MAX_CHILDREN],
            len: 0,
        }
    }

    fn push(&mut self, index: usize) {
        debug_assert!((self.len as usize) < MAX_CHILDREN);
        self.indices[self.len as usize] = index;
        self.len += 1;
    }

    fn len(&self) -> usize {
        self.len as usize
    }

    fn as_slice(&self) -> &[usize] {
        &self.indices[..self.len as usize]
    }
}

impl<U> BoundsTree<U>
where
    U: Clone
        + Debug
        + PartialEq
        + PartialOrd
        + Add<U, Output = U>
        + Sub<Output = U>
        + Half
        + Default,
{
    /// Clears all nodes from the tree.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.root = None;
        self.max_leaf = None;
        self.floor = 0;
        self.open.clear();
        self.reserved.clear();
        self.insert_path.clear();
        self.search_stack.clear();
    }

    /// Opens a layer that has to be drawn as a whole: returns an ordering above everything
    /// inserted so far, and holds every insert above it until [`BoundsTree::close`].
    pub fn open(&mut self) -> u32 {
        let start = self.highest() + 1;
        self.open.push((start, self.floor));
        self.floor = start;
        start
    }

    /// Closes the innermost open layer and returns the ordering it ends at. `footprint` is
    /// where the layer lands in the frame. A later insert that meets it goes above the layer,
    /// and one that does not keeps the ordering it would have had without the layer, so a
    /// layer opened for a press does not lift the rest of the frame over what it was drawn on.
    pub fn close(&mut self, footprint: Bounds<U>) -> u32 {
        let end = self.highest() + 1;
        let Some((start, floor)) = self.open.pop() else {
            return end;
        };
        self.floor = floor;
        // A layer clipped away lands nowhere, and a flat footprint would still meet whatever
        // straddles it.
        if !footprint.is_empty() {
            // Its kind sorts last, so nothing shares the ordering the layer ends at.
            let key = Key {
                ordering: end,
                kind: PrimitiveKind::Surface,
            };
            let leaf = self.insert_leaf(footprint, key);
            self.max_leaf = Some(leaf);
        }
        // Ranges closed inside this one start after it, so they sit at the end, in order.
        let inner = self.reserved.partition_point(|(from, _)| *from < start);
        self.reserved.truncate(inner);
        // Layers closed back to back hold one run of orderings, so an insert is lifted past all
        // of them at once.
        match self.reserved.last_mut() {
            Some((_, last)) if *last + 1 == start => *last = end,
            _ => self.reserved.push((start, end)),
        }
        end
    }

    fn highest(&self) -> u32 {
        self.max_leaf
            .map(|index| self.nodes[index].max_key.ordering)
            .unwrap_or_default()
            .max(self.floor)
            .max(self.reserved.last().map_or(0, |(_, end)| *end))
    }

    /// Moves an ordering that landed in a closed layer's range just past it, since nothing
    /// outside a layer may be drawn among its contents.
    fn lifted(&self, mut ordering: u32) -> u32 {
        loop {
            let after = self
                .reserved
                .partition_point(|(start, _)| *start <= ordering);
            match after.checked_sub(1).map(|index| self.reserved[index]) {
                Some((_, end)) if ordering <= end => ordering = end + 1,
                _ => return ordering,
            }
        }
    }

    /// Inserts bounds of the given kind into the tree and returns its assigned
    /// ordering.
    ///
    /// The ordering is one greater than the maximum ordering of any existing
    /// bounds that intersect with the new bounds. The new bounds share that
    /// maximum ordering instead if every intersecting bounds there has a kind
    /// of at most `max_shared_kind`, since those are drawn first. Inside an open
    /// layer the ordering stays above the layer's start, and an ordering that lands
    /// in a closed layer's range is lifted past it.
    pub fn insert(
        &mut self,
        new_bounds: Bounds<U>,
        kind: PrimitiveKind,
        max_shared_kind: PrimitiveKind,
    ) -> u32 {
        // Nothing at or below an open layer's floor counts, and nothing shares the floor.
        let ordering = match self.find_max_key(&new_bounds) {
            Some(max_key) if max_key.ordering > self.floor && max_key.kind <= max_shared_kind => {
                max_key.ordering
            }
            Some(max_key) if max_key.ordering > self.floor => max_key.ordering + 1,
            _ => self.floor + 1,
        };
        let ordering = self.lifted(ordering);
        let key = Key { ordering, kind };

        // Insert the new leaf
        let new_leaf_idx = self.insert_leaf(new_bounds, key);

        // Update max_leaf tracking
        self.max_leaf = match self.max_leaf {
            None => Some(new_leaf_idx),
            Some(old_idx) if self.nodes[old_idx].max_key < key => Some(new_leaf_idx),
            some => some,
        };

        ordering
    }

    /// Inserts bounds of the given kind with the ordering [`BoundsTree::insert`] gave them in an
    /// earlier frame that told the tree the same things up to this point. The tree ends up as
    /// `insert` would leave it, without searching for the ordering.
    pub fn insert_known(&mut self, new_bounds: Bounds<U>, ordering: u32, kind: PrimitiveKind) {
        #[cfg(test)]
        {
            self.known_inserts += 1;
        }
        let key = Key { ordering, kind };
        let new_leaf_idx = self.insert_leaf(new_bounds, key);
        self.max_leaf = match self.max_leaf {
            None => Some(new_leaf_idx),
            Some(old_idx) if self.nodes[old_idx].max_key < key => Some(new_leaf_idx),
            some => some,
        };
    }

    /// Finds the maximum key among all bounds that intersect with the query.
    fn find_max_key(&mut self, query: &Bounds<U>) -> Option<Key> {
        #[cfg(test)]
        {
            self.searches += 1;
        }
        let root_idx = self.root?;

        // Fast path: check if the max-key leaf intersects
        if let Some(max_idx) = self.max_leaf {
            let max_node = &self.nodes[max_idx];
            if query.intersects(&max_node.bounds) {
                return Some(max_node.max_key);
            }
        }

        // Slow path: search the tree
        self.search_stack.clear();
        self.search_stack.push(NonNull::from(&self.nodes[root_idx]));

        // `None` compares below every key.
        let mut max_found = None;

        while let Some(node) = self.search_stack.pop() {
            // SAFETY: `node` is guaranteed to be valid as the `nodes` stack is unmodified in this function
            // and the `search_stack` only contains pointers from this function call.
            let node = unsafe { node.as_ref() };

            // Pruning: skip if this subtree can't improve our result
            if Some(node.max_key) <= max_found {
                continue;
            }

            // Spatial pruning: skip if bounds don't intersect
            if !query.intersects(&node.bounds) {
                continue;
            }

            match &node.kind {
                NodeKind::Leaf { key } => {
                    max_found = cmp::max(max_found, Some(*key));
                }
                NodeKind::Internal { children } => {
                    // Children are maintained with highest max_key at the end.
                    // Push in forward order to highest (last) is popped first.
                    self.search_stack.extend(
                        children
                            .as_slice()
                            .iter()
                            .map(|&child_idx| &self.nodes[child_idx])
                            .filter(|node| Some(node.max_key) > max_found)
                            .map(NonNull::from),
                    );
                }
            }
        }

        max_found
    }

    /// Inserts a leaf node with the given bounds and key.
    /// Returns the index of the new leaf.
    fn insert_leaf(&mut self, bounds: Bounds<U>, key: Key) -> usize {
        let new_leaf_idx = self.nodes.len();
        self.nodes.push(Node {
            bounds: bounds.clone(),
            max_key: key,
            kind: NodeKind::Leaf { key },
        });

        let Some(root_idx) = self.root else {
            // Tree is empty, new leaf becomes root
            self.root = Some(new_leaf_idx);
            return new_leaf_idx;
        };

        // If root is a leaf, create internal node with both
        if matches!(self.nodes[root_idx].kind, NodeKind::Leaf { .. }) {
            let root_bounds = self.nodes[root_idx].bounds.clone();
            let root_key = self.nodes[root_idx].max_key;

            let mut children = NodeChildren::new();
            // Max end invariant
            if key > root_key {
                children.push(root_idx);
                children.push(new_leaf_idx);
            } else {
                children.push(new_leaf_idx);
                children.push(root_idx);
            }

            let new_root_idx = self.nodes.len();
            self.nodes.push(Node {
                bounds: root_bounds.union(&bounds),
                max_key: cmp::max(root_key, key),
                kind: NodeKind::Internal { children },
            });
            self.root = Some(new_root_idx);
            return new_leaf_idx;
        }

        // Descend to find the best internal node to insert into
        self.insert_path.clear();
        let mut current_idx = root_idx;

        loop {
            let current = &self.nodes[current_idx];
            let NodeKind::Internal { children } = &current.kind else {
                unreachable!("Should only traverse internal nodes");
            };

            self.insert_path.push(current_idx);

            // Find the best child to descend into
            let mut best_child_idx = children.as_slice()[0];
            let mut best_child_pos = 0;
            let mut best_cost = bounds
                .union(&self.nodes[best_child_idx].bounds)
                .half_perimeter();

            for (pos, &child_idx) in children.as_slice().iter().enumerate().skip(1) {
                let cost = bounds.union(&self.nodes[child_idx].bounds).half_perimeter();
                if cost < best_cost {
                    best_cost = cost;
                    best_child_idx = child_idx;
                    best_child_pos = pos;
                }
            }

            // Check if best child is a leaf or internal
            if matches!(self.nodes[best_child_idx].kind, NodeKind::Leaf { .. }) {
                // Best child is a leaf. Check if current node has room for another child.
                if children.len() < MAX_CHILDREN {
                    // Add new leaf directly to this node
                    let node = &mut self.nodes[current_idx];

                    if let NodeKind::Internal { children } = &mut node.kind {
                        children.push(new_leaf_idx);
                        // Swap new leaf only if it has the highest max_key
                        if key <= node.max_key {
                            let last = children.len() - 1;
                            children.indices.swap(last - 1, last);
                        }
                    }

                    node.bounds = node.bounds.union(&bounds);
                    node.max_key = cmp::max(node.max_key, key);
                    break;
                } else {
                    // Node is full, create new internal with [best_leaf, new_leaf]
                    let sibling_bounds = self.nodes[best_child_idx].bounds.clone();
                    let sibling_key = self.nodes[best_child_idx].max_key;

                    let mut new_children = NodeChildren::new();
                    // Max end invariant
                    if key > sibling_key {
                        new_children.push(best_child_idx);
                        new_children.push(new_leaf_idx);
                    } else {
                        new_children.push(new_leaf_idx);
                        new_children.push(best_child_idx);
                    }

                    let new_internal_idx = self.nodes.len();
                    let new_internal_max_key = cmp::max(sibling_key, key);
                    self.nodes.push(Node {
                        bounds: sibling_bounds.union(&bounds),
                        max_key: new_internal_max_key,
                        kind: NodeKind::Internal {
                            children: new_children,
                        },
                    });

                    // Replace the leaf with the new internal in parent
                    let parent = &mut self.nodes[current_idx];
                    if let NodeKind::Internal { children } = &mut parent.kind {
                        let children_len = children.len();

                        children.indices[best_child_pos] = new_internal_idx;

                        // If new internal has highest max_key, swap it to the end
                        // to maintain sorting invariant
                        if new_internal_max_key > parent.max_key {
                            children.indices.swap(best_child_pos, children_len - 1);
                        }
                    }
                    break;
                }
            } else {
                // Best child is internal, continue descent
                current_idx = best_child_idx;
            }
        }

        // Propagate bounds and max_key updates up the tree
        let mut updated_child_idx = None;
        for &node_idx in self.insert_path.iter().rev() {
            let node = &mut self.nodes[node_idx];
            node.bounds = node.bounds.union(&bounds);

            if node.max_key < key {
                node.max_key = key;

                // Swap updated child to end (skip first iteration since the invariant is already handled by previous cases)
                if let Some(child_idx) = updated_child_idx {
                    if let NodeKind::Internal { children } = &mut node.kind {
                        if let Some(pos) = children.as_slice().iter().position(|&c| c == child_idx)
                        {
                            let last = children.len() - 1;
                            if pos != last {
                                children.indices.swap(pos, last);
                            }
                        }
                    }
                }
            }

            updated_child_idx = Some(node_idx);
        }

        new_leaf_idx
    }
}

impl<U> Default for BoundsTree<U>
where
    U: Clone + Debug + Default + PartialEq,
{
    fn default() -> Self {
        BoundsTree {
            nodes: Vec::new(),
            root: None,
            max_leaf: None,
            floor: 0,
            open: Vec::new(),
            reserved: Vec::new(),
            insert_path: Vec::new(),
            search_stack: Vec::new(),
            #[cfg(test)]
            searches: 0,
            #[cfg(test)]
            known_inserts: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Bounds, Point, Size};
    use rand::{Rng, SeedableRng};

    #[test]
    fn test_insert() {
        let mut tree = BoundsTree::<f32>::default();
        let (quad, shadow) = (PrimitiveKind::Quad, PrimitiveKind::Shadow);
        let bounds1 = Bounds {
            origin: Point { x: 0.0, y: 0.0 },
            size: Size {
                width: 10.0,
                height: 10.0,
            },
        };
        let bounds2 = Bounds {
            origin: Point { x: 5.0, y: 5.0 },
            size: Size {
                width: 10.0,
                height: 10.0,
            },
        };
        let bounds3 = Bounds {
            origin: Point { x: 10.0, y: 10.0 },
            size: Size {
                width: 10.0,
                height: 10.0,
            },
        };

        // Insert the bounds into the tree and verify the order is correct
        assert_eq!(tree.insert(bounds1, quad, shadow), 1);
        assert_eq!(tree.insert(bounds2, quad, shadow), 2);
        assert_eq!(tree.insert(bounds3, quad, shadow), 3);

        // Insert non-overlapping bounds and verify they can reuse orders
        let bounds4 = Bounds {
            origin: Point { x: 20.0, y: 20.0 },
            size: Size {
                width: 10.0,
                height: 10.0,
            },
        };
        let bounds5 = Bounds {
            origin: Point { x: 40.0, y: 40.0 },
            size: Size {
                width: 10.0,
                height: 10.0,
            },
        };
        let bounds6 = Bounds {
            origin: Point { x: 25.0, y: 25.0 },
            size: Size {
                width: 10.0,
                height: 10.0,
            },
        };
        assert_eq!(tree.insert(bounds4, quad, shadow), 1); // bounds4 does not overlap with bounds1, bounds2, or bounds3
        assert_eq!(tree.insert(bounds5, quad, shadow), 1); // bounds5 does not overlap with any other bounds
        assert_eq!(tree.insert(bounds6, quad, shadow), 2); // bounds6 overlaps with bounds4, so it should have a different order
    }

    #[test]
    fn test_random_iterations() {
        let max_bounds = 100;
        for seed in 1..=1000 {
            let mut tree = BoundsTree::default();
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed as u64);
            let mut expected_quads: Vec<(Bounds<f32>, u32, PrimitiveKind)> = Vec::new();
            let kinds = [
                PrimitiveKind::Shadow,
                PrimitiveKind::Quad,
                PrimitiveKind::Path,
                PrimitiveKind::Underline,
            ];

            // Insert a random number of random AABBs into the tree.
            let num_bounds = rng.random_range(1..=max_bounds);
            for _ in 0..num_bounds {
                let min_x: f32 = rng.random_range(-100.0..100.0);
                let min_y: f32 = rng.random_range(-100.0..100.0);
                let width: f32 = rng.random_range(0.0..50.0);
                let height: f32 = rng.random_range(0.0..50.0);
                let bounds = Bounds {
                    origin: Point { x: min_x, y: min_y },
                    size: Size { width, height },
                };
                let kind = kinds[rng.random_range(0..kinds.len())];
                let max_shared_kind = kinds[rng.random_range(0..kinds.len())];

                let max_intersecting = expected_quads
                    .iter()
                    .filter(|quad| quad.0.intersects(&bounds))
                    .map(|quad| (quad.1, quad.2))
                    .max();
                let expected_ordering = match max_intersecting {
                    None => 1,
                    Some((ordering, max_kind)) if max_kind <= max_shared_kind => ordering,
                    Some((ordering, _)) => ordering + 1,
                };
                expected_quads.push((bounds, expected_ordering, kind));

                let actual_ordering = tree.insert(bounds, kind, max_shared_kind);
                assert_eq!(actual_ordering, expected_ordering);
            }
        }
    }
}
