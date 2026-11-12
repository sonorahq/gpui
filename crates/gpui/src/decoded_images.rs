//! Decoded pictures `img` keeps loaded, held to a byte budget once an app sets one: past it,
//! the least recently drawn pictures that no window shows are let go, their atlas tiles with
//! them, and decode again if they are drawn later.

use std::any::TypeId;
use std::sync::Arc;

use collections::{FxHashSet, IndexMap};

use crate::{App, ImageId, RenderImage, Window};

#[derive(Default)]
pub(crate) struct DecodedImages {
    /// By asset key, least recently drawn first.
    held: IndexMap<(TypeId, u64), Arc<RenderImage>>,
    bytes: usize,
    budget: Option<usize>,
}

/// What a picture costs once drawn: its pixels, and the texture of the same size a window
/// draws them from.
fn size_of(image: &RenderImage) -> usize {
    let pixels: usize = (0..image.frame_count())
        .filter_map(|frame| image.as_bytes(frame))
        .map(<[u8]>::len)
        .sum();
    pixels * 2
}

impl DecodedImages {
    pub(crate) fn drawn(&mut self, key: (TypeId, u64), image: &Arc<RenderImage>) {
        if self.budget.is_none() {
            return;
        }
        match self.held.shift_remove(&key) {
            Some(held) => {
                self.held.insert(key, held);
            }
            None => {
                self.bytes += size_of(image);
                self.held.insert(key, image.clone());
            }
        }
    }
}

impl App {
    /// Holds decoded pictures, with the textures drawn from them, to `bytes` in all: once a
    /// frame leaves more than that loaded, the least recently drawn that no window shows are
    /// dropped until it fits.
    pub fn set_decoded_image_budget(&mut self, bytes: usize) {
        self.decoded_images.budget = Some(bytes);
    }

    /// Counts a picture an app's own [`crate::Asset`] decoded, and drew this frame, against the
    /// budget, as `img` counts its own: past the budget, it is let go once no window shows it,
    /// and [`crate::Window::use_asset`] decodes it again when it is drawn later.
    pub fn decoded_image_drawn<A: crate::Asset>(
        &mut self,
        source: &A::Source,
        image: &Arc<RenderImage>,
    ) {
        let key = (TypeId::of::<A>(), crate::hash(source));
        self.decoded_images.drawn(key, image);
    }

    /// Lets go of pictures past the budget, once `current` has drawn its frame.
    pub(crate) fn trim_decoded_images(&mut self, current: &mut Window) {
        let Some(budget) = self.decoded_images.budget else {
            return;
        };
        if self.decoded_images.bytes <= budget {
            return;
        }
        let mut shown: FxHashSet<ImageId> = current.shown_images();
        for window in self.windows.values().flatten() {
            shown.extend(window.shown_images());
        }
        let mut bytes = self.decoded_images.bytes;
        let mut dropped = Vec::new();
        self.decoded_images.held.retain(|key, image| {
            if bytes <= budget || shown.contains(&image.id) {
                return true;
            }
            bytes -= size_of(image);
            dropped.push((*key, image.clone()));
            false
        });
        self.decoded_images.bytes = bytes;
        for (key, image) in dropped {
            self.loading_assets.remove(&key);
            self.drop_image(image, Some(current));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::Arc;

    use crate::{
        Context, Image, ImageFormat, ImageSource, IntoElement, ParentElement as _, Render,
        Styled as _, TestAppContext, Window, div, img, px,
    };

    /// A 16 by 16 picture, 1024 bytes once decoded, in a colour no other one shares.
    fn picture(shade: u8) -> Arc<Image> {
        let pixels = image::RgbaImage::from_pixel(16, 16, image::Rgba([shade, 0, 0, 255]));
        let mut png = Vec::new();
        pixels
            .write_to(&mut Cursor::new(&mut png), image::ImageFormat::Png)
            .expect("a png");
        Arc::new(Image::from_bytes(ImageFormat::Png, png))
    }

    struct Pictures(Vec<Arc<Image>>);

    impl Render for Pictures {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            div().size_full().children(
                self.0
                    .iter()
                    .map(|picture| img(ImageSource::Image(picture.clone())).size(px(16.))),
            )
        }
    }

    #[crate::test]
    fn a_picture_no_longer_drawn_is_let_go_past_the_budget(cx: &mut TestAppContext) {
        cx.update(|cx| cx.set_decoded_image_budget(1500));
        let (first, second) = (picture(1), picture(2));
        let (view, cx) = cx.add_window_view(|_, _| Pictures(vec![first.clone()]));
        cx.run_until_parked();
        assert!(cx.update(|_, cx| ImageSource::Image(first.clone()).is_asset_cached(cx)));

        view.update(cx, |pictures, cx| {
            pictures.0 = vec![second.clone()];
            cx.notify();
        });
        cx.run_until_parked();
        cx.update(|_, cx| {
            assert!(!ImageSource::Image(first.clone()).is_asset_cached(cx));
            assert!(ImageSource::Image(second.clone()).is_asset_cached(cx));
        });
    }

    /// An app's own loader, which decodes 16 by 16 pictures of one shade.
    enum Shades {}

    impl crate::Asset for Shades {
        type Source = u8;
        type Output = Result<Arc<crate::RenderImage>, crate::ImageCacheError>;

        fn load(
            shade: u8,
            _: &mut crate::App,
        ) -> impl std::future::Future<Output = Self::Output> + Send + 'static {
            async move {
                let pixels = image::RgbaImage::from_pixel(16, 16, image::Rgba([0, 0, shade, 255]));
                Ok(Arc::new(crate::RenderImage::new([image::Frame::new(
                    pixels,
                )])))
            }
        }
    }

    struct Shaded(u8);

    impl Render for Shaded {
        fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
            let shade = self.0;
            img(ImageSource::Custom(Arc::new(move |window, cx| {
                let image = window.use_asset::<Shades>(&shade, cx);
                if let Some(Ok(image)) = &image {
                    cx.decoded_image_drawn::<Shades>(&shade, image);
                }
                image
            })))
            .size(px(16.))
        }
    }

    #[crate::test]
    fn an_apps_own_picture_counts_against_the_budget(cx: &mut TestAppContext) {
        cx.update(|cx| cx.set_decoded_image_budget(1500));
        let (view, cx) = cx.add_window_view(|_, _| Shaded(1));
        cx.run_until_parked();
        view.update(cx, |_, cx| cx.notify());
        cx.run_until_parked();
        assert!(cx.update(|_, cx| cx.has_asset::<Shades>(&1)));

        view.update(cx, |shaded, cx| {
            shaded.0 = 2;
            cx.notify();
        });
        cx.run_until_parked();
        view.update(cx, |_, cx| cx.notify());
        cx.run_until_parked();
        cx.update(|_, cx| {
            assert!(
                !cx.has_asset::<Shades>(&1),
                "the picture no longer drawn is let go"
            );
            assert!(cx.has_asset::<Shades>(&2));
        });
    }

    #[crate::test]
    fn a_picture_on_screen_is_kept_over_the_budget(cx: &mut TestAppContext) {
        cx.update(|cx| cx.set_decoded_image_budget(0));
        let (first, second) = (picture(1), picture(2));
        let (view, cx) = cx.add_window_view(|_, _| Pictures(vec![first.clone(), second.clone()]));
        cx.run_until_parked();
        view.update(cx, |_, cx| cx.notify());
        cx.run_until_parked();
        cx.update(|_, cx| {
            assert!(ImageSource::Image(first.clone()).is_asset_cached(cx));
            assert!(ImageSource::Image(second.clone()).is_asset_cached(cx));
        });
    }
}
