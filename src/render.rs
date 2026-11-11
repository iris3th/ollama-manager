use eframe::egui;
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};

/// Render Markdown with a warm, high-contrast palette that remains readable
/// on the parchment conversation surface.
pub(crate) fn render_message(
    ui: &mut egui::Ui,
    _ctx: &egui::Context,
    content: &str,
) {
    let mut cache = CommonMarkCache::default();
    ui.scope(|ui| {
        let visuals = ui.visuals_mut();
        visuals.override_text_color = Some(egui::Color32::from_rgb(60, 47, 35));
        visuals.widgets.noninteractive.fg_stroke.color = egui::Color32::from_rgb(60, 47, 35);
        visuals.widgets.inactive.fg_stroke.color = egui::Color32::from_rgb(60, 47, 35);
        visuals.widgets.hovered.fg_stroke.color = egui::Color32::from_rgb(60, 47, 35);
        visuals.code_bg_color = egui::Color32::from_rgb(237, 228, 212);
        visuals.hyperlink_color = egui::Color32::from_rgb(145, 91, 39);
        CommonMarkViewer::new().show(ui, &mut cache, content);
    });
}
