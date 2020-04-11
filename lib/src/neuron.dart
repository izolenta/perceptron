import 'package:meta/meta.dart';
import 'package:perceptron/src/activation_functions/activation_function.dart';

class Neuron {
  final int layer;
  final int number;

  double _value = 0;
  double get value => _value;

  ActivationFunction activationFunction;

  Neuron({@required this.layer, @required this.number, @required this.activationFunction});

  void initNeuron() => _value = 0;

  void setExplicitValue(double value) {
    assert(layer == 0, 'You shouldn\'t set explicit values to neurons in layers different than input!');
    _value = value;
  }

  void addWeightedValue(double weight) => _value += weight;

  void sealValue() {
    _value = activationFunction.process(_value);
  }

}