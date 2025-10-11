#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QPushButton>
#include <QLabel>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QGroupBox>
#include <QTextEdit>
#include <QTimer>

#include "Tensor.h"
#include "Network.h"
#include "Training.h"
#include "Loss.h"
#include "Optimizer.h"
#include "networkvisualizationwidget.h"

class TrainingWidget : public QWidget
{
    Q_OBJECT

public:
    explicit TrainingWidget(QWidget *parent = nullptr);

signals:
    void statusUpdated(const QString &message);

private slots:
    void startTraining();
    void stopTraining();
    void updateTraining();
    void updateVisualization();

private:
    void setupUI();
    void setupNetwork();
    void setupTraining();
    void addXORData();
    void addRandomData();

    nn::Network network;
    nn::MSELoss lossFunction;
    nn::SGD optimizer;

    // UI Elements
    QVBoxLayout *mainLayout;
    NetworkVisualizationWidget *networkViz;
    QTextEdit *logArea;
    QPushButton *startButton;
    QPushButton *stopButton;
    QSpinBox *epochsSpinBox;
    QDoubleSpinBox *learningRateSpinBox;
    QSpinBox *hiddenSizeSpinBox;

    // Training variables
    QTimer *trainingTimer;
    bool isTraining;
    int currentEpoch;
    std::vector<std::pair<nn::Tensor, nn::Tensor>> trainingData;
    
    // Visualization variables
    std::vector<std::vector<double>> networkWeights;
    std::vector<std::vector<double>> networkActivations;
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);

private:
    TrainingWidget *trainingWidget;
};

#endif // MAINWINDOW_H