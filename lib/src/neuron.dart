import 'package:meta/meta.dart';
import 'package:perceptron/src/activation_functions/activation_function.dart';
import 'package:perceptron/src/activation_functions/training_neuron_description.dart';

class Neuron {

  final int layer;
  final int number;

  final prevLayerValues = <TrainingNeuronDescription>[];

  double _value = 0;
  double get value => _value;

  double _unsealedValue = 0;
  double get unsealedValue => _unsealedValue;

  double _error = 0;
  double get error => _error;

  ActivationFunction activationFunction;

  Neuron({@required this.layer, @required this.number, @required this.activationFunction});

  void initNeuron() {
    _value = 0;
    _error = 0;
    _unsealedValue = 0;
    prevLayerValues.clear();
  }

  void setExplicitValue(double value) {
    assert(layer == 0, 'You shouldn\'t set explicit values to neurons in layers different than input!');
    _value = value;
  }

  void addErrorValue(double value) {
    _error += value;
  }

  void addWeightedValue(TrainingNeuronDescription description) {
    _value += description.neuronValue * description.synapseWeight;
    prevLayerValues.add(description);
  }
  void sealValue() {
    _unsealedValue = value;
    _value = activationFunction.process(_value);
  }
}