#include "mainwindow.h"
#include <QApplication>
#include <QPushButton>
#include <QLabel>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QTextEdit>
#include <QTimer>
#include <QMessageBox>
#include <QPainter>
#include <QStyleOption>
#include <QStyle>
#include <cmath>

TrainingWidget::TrainingWidget(QWidget *parent)
    : QWidget(parent), optimizer(0.1f), trainingTimer(new QTimer(this)), 
      isTraining(false), currentEpoch(0)
{
    setupUI();
    setupNetwork();
    setupTraining();
    addXORData();
}

void TrainingWidget::setupUI()
{
    mainLayout = new QVBoxLayout(this);

    // Controls group
    QGroupBox *controlsGroup = new QGroupBox("Training Controls");
    QHBoxLayout *controlsLayout = new QHBoxLayout(controlsGroup);

    epochsSpinBox = new QSpinBox();
    epochsSpinBox->setRange(1, 10000);
    epochsSpinBox->setValue(100);
    epochsSpinBox->setSuffix(" epochs");

    learningRateSpinBox = new QDoubleSpinBox();
    learningRateSpinBox->setRange(0.001, 1.0);
    learningRateSpinBox->setValue(0.1);
    learningRateSpinBox->setSingleStep(0.01);

    hiddenSizeSpinBox = new QSpinBox();
    hiddenSizeSpinBox->setRange(1, 100);
    hiddenSizeSpinBox->setValue(4);
    hiddenSizeSpinBox->setSuffix(" neurons");

    startButton = new QPushButton("Start Training");
    stopButton = new QPushButton("Stop Training");

    controlsLayout->addWidget(new QLabel("Epochs:"));
    controlsLayout->addWidget(epochsSpinBox);
    controlsLayout->addWidget(new QLabel("Learning Rate:"));
    controlsLayout->addWidget(learningRateSpinBox);
    controlsLayout->addWidget(new QLabel("Hidden Size:"));
    controlsLayout->addWidget(hiddenSizeSpinBox);
    controlsLayout->addWidget(startButton);
    controlsLayout->addWidget(stopButton);
    controlsLayout->addStretch();

    // Network visualization
    QGroupBox *visualizationGroup = new QGroupBox("Network Visualization");
    QVBoxLayout *vizLayout = new QVBoxLayout(visualizationGroup);

    networkViz = new NetworkVisualizationWidget();
    vizLayout->addWidget(networkViz);

    // Log area
    QGroupBox *logGroup = new QGroupBox("Training Log");
    QVBoxLayout *logLayout = new QVBoxLayout(logGroup);
    
    logArea = new QTextEdit();
    logArea->setReadOnly(true);
    logLayout->addWidget(logArea);

    // Add all to main layout
    mainLayout->addWidget(controlsGroup);
    mainLayout->addWidget(visualizationGroup);
    mainLayout->addWidget(logGroup);

    // Connect signals
    connect(startButton, &QPushButton::clicked, this, &TrainingWidget::startTraining);
    connect(stopButton, &QPushButton::clicked, this, &TrainingWidget::stopTraining);
    connect(trainingTimer, &QTimer::timeout, this, &TrainingWidget::updateTraining);
}

void TrainingWidget::setupNetwork()
{
    try {
        // Clear existing layers
        auto& existingLayers = network.get_layers();
        for (auto* layer : existingLayers) {
            delete layer;
        }
        
        // Create a simple network: 2 -> hidden -> 1
        nn::Linear* inputLayer = new nn::Linear(2, hiddenSizeSpinBox->value());
        nn::Sigmoid* activation1 = new nn::Sigmoid();
        nn::Linear* outputLayer = new nn::Linear(hiddenSizeSpinBox->value(), 1);
        nn::Sigmoid* outputActivation = new nn::Sigmoid();
        
        network.add_layer(inputLayer);
        network.add_layer(activation1);
        network.add_layer(outputLayer);
        network.add_layer(outputActivation);
        
        logArea->append("Network created: 2 -> " + QString::number(hiddenSizeSpinBox->value()) + " -> 1");
    } catch (const std::exception& e) {
        logArea->append("Error creating network: " + QString(e.what()));
    }
}

void TrainingWidget::setupTraining()
{
    optimizer = nn::SGD(learningRateSpinBox->value());
}

void TrainingWidget::addXORData()
{
    trainingData.clear();
    
    // XOR training data
    trainingData.push_back(std::make_pair(
        nn::Tensor(std::vector<float>{0.0f, 0.0f}, std::vector<size_t>{2, 1}),
        nn::Tensor(std::vector<float>{0.0f}, std::vector<size_t>{1, 1})
    ));
    
    trainingData.push_back(std::make_pair(
        nn::Tensor(std::vector<float>{0.0f, 1.0f}, std::vector<size_t>{2, 1}),
        nn::Tensor(std::vector<float>{1.0f}, std::vector<size_t>{1, 1})
    ));
    
    trainingData.push_back(std::make_pair(
        nn::Tensor(std::vector<float>{1.0f, 0.0f}, std::vector<size_t>{2, 1}),
        nn::Tensor(std::vector<float>{1.0f}, std::vector<size_t>{1, 1})
    ));
    
    trainingData.push_back(std::make_pair(
        nn::Tensor(std::vector<float>{1.0f, 1.0f}, std::vector<size_t>{2, 1}),
        nn::Tensor(std::vector<float>{0.0f}, std::vector<size_t>{1, 1})
    ));
    
    logArea->append("XOR training data loaded (4 samples)");
}

void TrainingWidget::startTraining()
{
    if (isTraining) return;
    
    isTraining = true;
    currentEpoch = 0;
    startButton->setEnabled(false);
    stopButton->setEnabled(true);
    logArea->append("Starting training...");
    
    setupNetwork();
    setupTraining();
    
    trainingTimer->start(50); // Update every 50ms
}

void TrainingWidget::stopTraining()
{
    isTraining = false;
    trainingTimer->stop();
    startButton->setEnabled(true);
    stopButton->setEnabled(false);
    logArea->append("Training stopped.");
}

void TrainingWidget::updateTraining()
{
    if (!isTraining || currentEpoch >= epochsSpinBox->value()) {
        stopTraining();
        return;
    }
    
    // Train on all data points
    float totalLoss = 0.0f;
    for (auto& dataPair : trainingData) {
        float loss = network.train(dataPair.first, dataPair.second, lossFunction, optimizer);
        totalLoss += loss;
    }
    
    float avgLoss = totalLoss / trainingData.size();
    
    if (currentEpoch % 10 == 0 || currentEpoch == 1) { // Log every 10 epochs or first epoch
        logArea->append(QString("Epoch %1, Average Loss: %2")
                       .arg(currentEpoch)
                       .arg(avgLoss, 0, 'f', 6));
    }
    
    currentEpoch++;
    
    // Update visualization periodically
    if (currentEpoch % 5 == 0) {
        updateVisualization();
    }
}

void TrainingWidget::updateVisualization()
{
    // In a real implementation, we would extract weights and activations from the network
    // For now, we'll just trigger a redraw
    emit statusUpdated("Network updated at epoch " + QString::number(currentEpoch));
}

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
{
    setWindowTitle("Neural Network Visualizer");
    resize(1200, 800);
    
    trainingWidget = new TrainingWidget(this);
    setCentralWidget(trainingWidget);
}
#include "moc_mainwindow.cpp"
