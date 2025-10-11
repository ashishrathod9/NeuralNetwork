#ifndef NETWORKVISUALIZATIONWIDGET_H
#define NETWORKVISUALIZATIONWIDGET_H

#include <QWidget>
#include <QPainter>
#include <QStyleOption>
#include <QStyle>
#include <QColor>
#include <QPen>
#include <QPoint>
#include <QVector>
#include <QString>

class NetworkVisualizationWidget : public QWidget
{
    Q_OBJECT

public:
    explicit NetworkVisualizationWidget(QWidget *parent = nullptr) : QWidget(parent) {}

    void setNetworkData(const std::vector<std::vector<double>> &weights, 
                       const std::vector<std::vector<double>> &activations) {
        networkWeights = weights;
        networkActivations = activations;
        update();
    }

protected:
    void paintEvent(QPaintEvent *event) override {
        QStyleOption opt;
        opt.initFrom(this);
        QPainter p(this);
        style()->drawPrimitive(QStyle::PE_Widget, &opt, this);

        if (networkWeights.empty() || networkActivations.empty()) {
            // Draw a simple message if no data
            p.setPen(Qt::black);
            p.drawText(rect(), Qt::AlignCenter, "Network Visualization\n(Start Training to See Network)");
            return;
        }

        // Draw neurons and connections
        int layerCount = networkActivations.size();
        if (layerCount < 2) return;

        int widgetWidth = width();
        int widgetHeight = height();
        int neuronRadius = 15;

        // Calculate positions for each layer
        std::vector<std::vector<QPoint>> neuronPositions(layerCount);
        
        for (int layer = 0; layer < layerCount; ++layer) {
            int neuronCount = networkActivations[layer].size();
            int layerX = (widgetWidth / (layerCount + 1)) * (layer + 1);
            
            for (int neuron = 0; neuron < neuronCount; ++neuron) {
                int neuronY = (widgetHeight / (neuronCount + 1)) * (neuron + 1);
                neuronPositions[layer].push_back(QPoint(layerX, neuronY));
            }
        }

        // Draw connections between layers
        for (int layer = 0; layer < layerCount - 1; ++layer) {
            int nextLayer = layer + 1;
            for (int i = 0; i < networkActivations[layer].size(); ++i) {
                for (int j = 0; j < networkActivations[nextLayer].size(); ++j) {
                    // Get weight for this connection
                    double weight = 0;
                    if (layer < networkWeights.size() && 
                        i < networkWeights[layer].size() && 
                        j < networkActivations[nextLayer].size()) {
                        // Calculate index in flattened weights
                        weight = networkWeights[layer][i * networkActivations[nextLayer].size() + j];
                    }

                    // Set color based on weight value
                    QColor color;
                    if (weight > 0) {
                        color = QColor(0, static_cast<int>(255 * std::min(1.0, weight)), 0, 150);  // Green for positive
                    } else {
                        color = QColor(static_cast<int>(255 * std::min(1.0, -weight)), 0, 0, 150);  // Red for negative
                    }

                    p.setPen(QPen(color, 1 + static_cast<int>(std::abs(weight) * 3)));
                    p.drawLine(neuronPositions[layer][i], neuronPositions[nextLayer][j]);
                }
            }
        }

        // Draw neurons
        for (int layer = 0; layer < layerCount; ++layer) {
            for (int neuron = 0; neuron < networkActivations[layer].size(); ++neuron) {
                double activation = networkActivations[layer][neuron];
                // Adjust neuron color based on activation
                int brightness = 100 + static_cast<int>(100 * std::abs(activation));
                brightness = std::min(255, brightness); // Cap at 255
                QColor neuronColor(brightness, brightness, 200);
                
                p.setBrush(neuronColor);
                p.setPen(Qt::black);
                
                QPoint pos = neuronPositions[layer][neuron];
                p.drawEllipse(pos, neuronRadius, neuronRadius);
                
                // Draw activation value if it's meaningful
                if (layer == 0 || layer == layerCount - 1) { // Only label input/output layers
                    p.setPen(Qt::black);
                    QString label = QString::number(activation, 'f', 2);
                    p.drawText(pos.x() - 15, pos.y() + 5, label);
                }
            }
        }
    }

private:
    std::vector<std::vector<double>> networkWeights;
    std::vector<std::vector<double>> networkActivations;
};

#endif // NETWORKVISUALIZATIONWIDGET_H
#include "moc_networkvisualizationwidget.cpp"
