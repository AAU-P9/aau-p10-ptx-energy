// This file is part of the "Tracy" project, an extensive profiler for games and other applications.
// Source: https://github.com/wolfpld/tracy/blob/master/public/tracy/TracyCUDA.hpp

struct IncrementalRegression {
    using float_t = double;
    struct Parameters {
        float_t slope, intercept;
    };

    int n = 0;
    float_t x_mean = 0;
    float_t y_mean = 0;
    float_t x_svar = 0;
    float_t y_svar = 0;
    float_t xy_scov = 0;

    auto parameters() const {
        float_t slope = xy_scov / x_svar;
        float_t intercept = y_mean - slope * x_mean;
        return Parameters{ slope, intercept };
    }

    auto orthogonal() const {
        // NOTE(marcos): orthogonal regression is Deming regression with delta = 1
        float_t delta = float_t(1);   // delta = 1 -> orthogonal regression
        float_t k = y_svar - delta * x_svar;
        float_t slope = (k + sqrt(k * k + 4 * delta * xy_scov * xy_scov)) / (2 * xy_scov);
        float_t intercept = y_mean - slope * x_mean;
        return Parameters{ slope, intercept };
    }

    void addSample(float_t x, float_t y) {
        ++n;
        float_t x_mean_prev = x_mean;
        float_t y_mean_prev = y_mean;
        x_mean += (x - x_mean) / n;
        y_mean += (y - y_mean) / n;
        x_svar += (x - x_mean_prev) * (x - x_mean);
        y_svar += (y - y_mean_prev) * (y - y_mean);
        xy_scov += (x - x_mean_prev) * (y - y_mean);
    }
};

