#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>

struct IndicatorData {
    std::string name;
    std::vector<int> midterm;
    std::vector<int> final;
};

struct IndicatorResult {
    std::array<std::array<double, 4>, 4> transition_matrix{};
    std::array<double, 4> stationary{};
    std::array<double, 4> membership{};
};

std::vector<int> generate_ratings(std::mt19937 &gen, std::size_t count) {
    std::vector<int> ratings(count);
    std::discrete_distribution<int> dist({1, 2, 4, 6});
    for (auto &value : ratings) {
        value = dist(gen);
    }
    return ratings;
}

IndicatorData build_indicator(std::mt19937 &gen, const std::string &name) {
    return IndicatorData{name, generate_ratings(gen, 100), generate_ratings(gen, 100)};
}

std::array<std::array<double, 4>, 4> compute_transition_matrix(const std::vector<int> &midterm,
                                                              const std::vector<int> &final) {
    std::array<std::array<double, 4>, 4> matrix{};
    std::array<int, 4> row_totals{};

    const auto sample_count = std::min(midterm.size(), final.size());
    for (std::size_t i = 0; i < sample_count; ++i) {
        const int from = std::clamp(midterm[i], 0, 3);
        const int to = std::clamp(final[i], 0, 3);
        matrix[from][to] += 1.0;
        row_totals[from] += 1;
    }

    for (std::size_t row = 0; row < 4; ++row) {
        const double total = static_cast<double>(row_totals[row]);
        if (total == 0) {
            for (auto &cell : matrix[row]) {
                cell = 0.25;
            }
        } else {
            for (auto &cell : matrix[row]) {
                cell /= total;
            }
        }
    }
    return matrix;
}

std::array<double, 4> compute_stationary_distribution(
    const std::array<std::array<double, 4>, 4> &matrix) {
    std::array<double, 4> distribution{0.25, 0.25, 0.25, 0.25};
    std::array<double, 4> next{};

    constexpr double tolerance = 1e-9;
    for (int iteration = 0; iteration < 10'000; ++iteration) {
        for (std::size_t col = 0; col < 4; ++col) {
            next[col] = 0.0;
            for (std::size_t row = 0; row < 4; ++row) {
                next[col] += distribution[row] * matrix[row][col];
            }
        }
        const double delta = std::inner_product(next.begin(), next.end(), distribution.begin(), 0.0,
                                                std::plus<>(), [](double a, double b) {
                                                    const double diff = a - b;
                                                    return diff * diff;
                                                });
        distribution = next;
        if (delta < tolerance) {
            break;
        }
    }
    return distribution;
}

std::array<double, 4> compute_membership(const std::array<double, 4> &stationary) {
    // Fuzzy membership for four linguistic levels: Poor, Average, Good, Excellent.
    const std::array<std::array<double, 4>, 4> fuzzy_mapping{
        std::array<double, 4>{1.0, 0.0, 0.0, 0.0},
        std::array<double, 4>{0.25, 0.5, 0.25, 0.0},
        std::array<double, 4>{0.0, 0.25, 0.5, 0.25},
        std::array<double, 4>{0.0, 0.0, 0.3, 0.7},
    };

    std::array<double, 4> membership{};
    for (std::size_t level = 0; level < 4; ++level) {
        double value = 0.0;
        for (std::size_t state = 0; state < 4; ++state) {
            value += stationary[state] * fuzzy_mapping[state][level];
        }
        membership[level] = value;
    }
    return membership;
}

IndicatorResult evaluate_indicator(const IndicatorData &indicator) {
    auto transition_matrix = compute_transition_matrix(indicator.midterm, indicator.final);
    auto stationary = compute_stationary_distribution(transition_matrix);
    auto membership = compute_membership(stationary);
    return IndicatorResult{transition_matrix, stationary, membership};
}

std::vector<double> generate_importance_scores(std::mt19937 &gen, std::size_t count) {
    std::vector<double> scores(count);
    std::normal_distribution<double> distribution(7.5, 1.5);
    for (auto &score : scores) {
        score = std::clamp(distribution(gen), 1.0, 9.0);
    }
    return scores;
}

std::vector<double> derive_weights(const std::vector<double> &scores) {
    const double sum = std::accumulate(scores.begin(), scores.end(), 0.0);
    std::vector<double> weights(scores.size());
    for (std::size_t i = 0; i < scores.size(); ++i) {
        weights[i] = scores[i] / sum;
    }
    return weights;
}

void print_transition_matrix(const std::array<std::array<double, 4>, 4> &matrix) {
    std::cout << "  一步转移矩阵 (行=期中, 列=期末):\n";
    for (const auto &row : matrix) {
        for (const double value : row) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << value << ' ';
        }
        std::cout << '\n';
    }
}

void print_vector(const std::array<double, 4> &vec, const std::string &title) {
    std::cout << "  " << title << ": [";
    for (std::size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << vec[i];
        if (i + 1 < vec.size()) {
            std::cout << ", ";
        }
    }
    std::cout << "]\n";
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());

    const std::vector<std::string> indicator_names = {
        "教师素质",       "教学态度",       "教学内容",   "教学方法",
        "教学效果",       "考核公平性",     "学生参与度", "资源保障",
        "教学创新",       "实践应用"};

    std::vector<IndicatorData> indicators;
    indicators.reserve(indicator_names.size());
    for (const auto &name : indicator_names) {
        indicators.emplace_back(build_indicator(gen, name));
    }

    std::cout << "========== 随机生成的期中与期末教评数据 (各100条) ==========" << '\n';
    for (const auto &indicator : indicators) {
        std::cout << indicator.name << " -> 期中评分样本均值: "
                  << std::accumulate(indicator.midterm.begin(), indicator.midterm.end(), 0.0) /
                         indicator.midterm.size()
                  << ", 期末评分样本均值: "
                  << std::accumulate(indicator.final.begin(), indicator.final.end(), 0.0) /
                         indicator.final.size()
                  << '\n';
    }

    std::vector<IndicatorResult> results;
    results.reserve(indicators.size());

    std::cout << "\n========== 教学质量评价的逐步计算 ==========" << '\n';
    for (std::size_t idx = 0; idx < indicators.size(); ++idx) {
        results.emplace_back(evaluate_indicator(indicators[idx]));
        std::cout << "\n指标 " << indicators[idx].name << '\n';
        print_transition_matrix(results.back().transition_matrix);
        print_vector(results.back().stationary, "平稳分布");
        print_vector(results.back().membership, "隶属度向量 (差评/中评/良好/优秀)");
    }

    std::cout << "\n========== 指标权重计算 (求平均数法 + 层次分析法近似) =========="
              << '\n';
    const auto importance_scores = generate_importance_scores(gen, indicators.size());
    const auto weights = derive_weights(importance_scores);

    for (std::size_t i = 0; i < indicators.size(); ++i) {
        std::cout << std::setw(10) << indicators[i].name << " | 平均重要度: "
                  << std::fixed << std::setprecision(2) << importance_scores[i]
                  << " | 权重: " << std::fixed << std::setprecision(4) << weights[i] << '\n';
    }

    std::cout << "\n========== 模糊综合评价 ==========" << '\n';
    std::array<double, 4> evaluation_vector{0.0, 0.0, 0.0, 0.0};
    for (std::size_t idx = 0; idx < results.size(); ++idx) {
        for (std::size_t level = 0; level < 4; ++level) {
            evaluation_vector[level] += weights[idx] * results[idx].membership[level];
        }
    }

    print_vector(evaluation_vector, "教学质量评定向量 (差评/中评/良好/优秀)");

    const std::array<double, 4> score_scale{40.0, 60.0, 80.0, 100.0};
    double final_score = 0.0;
    for (std::size_t level = 0; level < 4; ++level) {
        final_score += evaluation_vector[level] * score_scale[level];
    }

    std::cout << "教学质量最终评分: " << std::fixed << std::setprecision(2) << final_score << " / 100"
              << '\n';

    return 0;
}
