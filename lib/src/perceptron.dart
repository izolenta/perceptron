import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:perceptron/src/activation_functions/activation_function.dart';
import 'package:perceptron/src/activation_functions/activation_function_type.dart';
import 'package:perceptron/src/activation_functions/bipolar_sigmoid.dart';
import 'package:perceptron/src/activation_functions/linear.dart';
import 'package:perceptron/src/activation_functions/sigmoid.dart';
import 'package:perceptron/src/neuron.dart';
import 'package:perceptron/src/synapse.dart';

class Perceptron {

  final _neurons = <Neuron>[];
  final _synapses = <Synapse>[];
  final Random _rand = Random();

  List<int> _netConfiguration;
  ActivationFunctionType _activationFunctionType;

  Perceptron(List<int> layers, [ActivationFunctionType activationFunctionType = ActivationFunctionType.sigmoid]) {
    _init(layers, activationFunctionType);
  }

  Perceptron.fromFile(String filename) {
    final Map<String, dynamic> loaded = jsonDecode(File(filename).readAsStringSync());
    final layers = List<int>.from(loaded['netConfiguration']);
    final activationFunctionType = loaded['activationFunction']?? ActivationFunctionType.sigmoid.value;
    final synapses = <Synapse>[];
    (loaded['synapses'] as List).forEach((next) =>
      synapses.add(Synapse(
        synapseLayer: next['layer'],
        originNeuron: next['origin'],
        destinationNeuron: next['destination'],
        weight: next['weight'],
      )));
    _init(layers, ActivationFunctionType.parse(activationFunctionType), synapses);
  }

  void _init(List<int> layers, ActivationFunctionType activationFunctionType, [List<Synapse> initialSynapses]) {
    assert(layers.length > 1, 'Neural network should have at least two layers');
    _activationFunctionType = activationFunctionType;
    final function = _getActivationFunction(activationFunctionType);
    _netConfiguration = layers;
    for (var i=0; i<layers.length; i++) {
      assert(layers[i] > 0, 'Neuron number in each layer should be positive');
      for (var j=0; j<layers[i]; j++) {
        _neurons.add(Neuron(layer: i, number: j, activationFunction: function));
      }
    }
    if (initialSynapses == null) {
      for (var i = 0; i < layers.length - 1; i++) {
        for (var j = 0; j < layers[i]; j++) {
          for (var k = 0; k < layers[i + 1]; k++) {
            _synapses.add(Synapse(
                synapseLayer: i,
                originNeuron: j,
                destinationNeuron: k,
                weight: _rand.nextDouble() - 0.5
            ));
          }
        }
      }
    }
    else {
      _synapses.addAll(initialSynapses);
    }
  }

  List<double> process(List<int> input) {
    final stopwatch = Stopwatch();
    stopwatch.start();
    _neurons.forEach((n) => n.initNeuron());
    for (var i=0; i<_netConfiguration.first; i++) {
      _getNeuron(0, i).setExplicitValue(input[i].toDouble());
    }
    for (var i=0; i<_netConfiguration.length-1; i++) {
      _processSynapseLayer(i);
    }
    final res = <double>[];
    for (var i=0; i<_netConfiguration.last; i++) {
      res.add(_getNeuron(_netConfiguration.length-1, i).value);
    }
    stopwatch.stop();
    print('processed in ${stopwatch.elapsedMilliseconds / 1000} sec');
    return res;
  }

  ActivationFunction _getActivationFunction(ActivationFunctionType activationFunctionType) {
    if (activationFunctionType == ActivationFunctionType.linear) {
      return Linear();
    }
    if (activationFunctionType == ActivationFunctionType.bipolarSigmoid) {
      return BipolarSigmoid();
    }
    return Sigmoid();
  }

  void _processSynapseLayer(int layer) {
    for (var i=0; i<_netConfiguration[layer]; i++) {
      final origin = _getNeuron(layer, i);
      for (var j=0; j<_netConfiguration[layer+1]; j++) {
        final destination = _getNeuron(layer+1, j);
        final synapse = _getSynapse(layer, i, j);
        destination.addWeightedValue(origin.value * synapse.weight);
      }
    }
    _neurons.where((n) => n.layer == layer + 1).forEach((f) => f.sealValue());
  }

  Neuron _getNeuron(int layer, int number) => _neurons.singleWhere((n) => n.layer == layer && n.number == number);
  Synapse _getSynapse(int layer, int origin, int destination) =>
      _synapses.singleWhere((s) => s.synapseLayer == layer && s.originNeuron == origin && s.destinationNeuron == destination);
  
  String toJson() {
    final map = <String, Object>{};
    map['netConfiguration'] = _netConfiguration;
    map['activationFunction'] = _activationFunctionType.value;
    map['synapses'] = _synapses.map((s) => s.toJson()).toList();
    return jsonEncode(map);
  }

  void saveToFile(String fileName) {
    var file = File(fileName);
    file.writeAsStringSync(toJson());
  }
}
