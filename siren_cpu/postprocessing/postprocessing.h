#include <image.h>

void PostProc(Image& img, std::vector<std::vector<Vector>>& color_map) {
    double max_value = 0;
    // Tone mapping
    for (int i = 0; i < img.Height(); ++i) {
        for (int j = 0; j < img.Width(); ++j) {
            max_value = std::max(max_value, color_map[i][j][0]);
            max_value = std::max(max_value, color_map[i][j][1]);
            max_value = std::max(max_value, color_map[i][j][2]);
        }
    }
    for (int i = 0; i < img.Height(); ++i) {
        for (int j = 0; j < img.Width(); ++j) {
            auto v_in = color_map[i][j];
            color_map[i][j] = (v_in * (1 + v_in / (max_value * max_value))) / (1 + v_in);
        }
    }
    // Gamma correction
    for (int i = 0; i < img.Height(); ++i) {
        for (int j = 0; j < img.Width(); ++j) {
            RGB color_int;
            color_int.r = static_cast<int>(255 * std::pow(color_map[i][j][0], 1.0 / 2.2));
            color_int.g = static_cast<int>(255 * std::pow(color_map[i][j][1], 1.0 / 2.2));
            color_int.b = static_cast<int>(255 * std::pow(color_map[i][j][2], 1.0 / 2.2));
            img.SetPixel(RGB{color_int.r, color_int.g, color_int.b}, i, j);
        }
    }
}
