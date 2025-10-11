#ifndef TRAINING_H
#define TRAINING_H

#include "Network.h"
#include "Data.h"
#include "Loss.h"
#include "Optimizer.h"
#include <vector>

namespace nn {

struct TrainingConfig {
    size_t epochs = 10;
    size_t batch_size = 32;
    float learning_rate = 0.01f;
    bool verbose = true;
    float validation_split = 0.2f;  // Fraction of data to use for validation

    TrainingConfig() = default;
    TrainingConfig(size_t epochs, size_t batch_size, float learning_rate,
                   bool verbose = true, float validation_split = 0.2f)
        : epochs(epochs),
          batch_size(batch_size),
          learning_rate(learning_rate),
          verbose(verbose),
          validation_split(validation_split)
    {}
};

class Trainer {
public:
    Trainer(TrainingConfig config = {});

    void train(Network& network,
               const std::vector<Tensor>& train_data,
               const std::vector<Tensor>& train_targets,
               Loss& loss_fn,
               Optimizer& optimizer);

    float validate(Network& network,
                   const std::vector<Tensor>& val_data,
                   const std::vector<Tensor>& val_targets,
                   Loss& loss_fn);

private:
    TrainingConfig config_;
};

} // namespace nn

#endif // TRAINING_H
