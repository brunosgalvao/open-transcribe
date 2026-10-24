use eframe::egui;
use open_transcribe::app;

fn main() -> eframe::Result {
    let icon = eframe::icon_data::from_png_bytes(include_bytes!("../assets/icon.png"))
        .expect("ícone embutido inválido");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1020.0, 790.0])
            .with_min_inner_size([840.0, 620.0])
            .with_title("Open Transcribe")
            .with_app_id("open-transcribe")
            .with_icon(icon),
        ..Default::default()
    };

    eframe::run_native(
        "Open Transcribe",
        options,
        Box::new(|cc| Ok(Box::new(app::App::new(cc)))),
    )
}
