import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:perceptron/src/activation_functions/activation_function.dart';
import 'package:perceptron/src/activation_functions/activation_function_type.dart';
import 'package:perceptron/src/activation_functions/bipolar_sigmoid.dart';
import 'package:perceptron/src/activation_functions/linear.dart';
import 'package:perceptron/src/activation_functions/sigmoid.dart';
import 'package:perceptron/src/activation_functions/training_neuron_description.dart';
import 'package:perceptron/src/neuron.dart';
import 'package:perceptron/src/synapse.dart';
import 'package:perceptron/src/training_data.dart';

import 'package:memoize/memoize.dart';

class Perceptron {

  static const alpha = 1;

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
    assert(layers.length == 3, 'This library supports neural networks with one input, one hidden and one output layers');
    _activationFunctionType = activationFunctionType;
    final function = _getActivationFunction(activationFunctionType);
    _netConfiguration = layers;
    for (var i=0; i<layers.length; i++) {
      assert(layers[i] > 0, 'Neuron number in each layer should be positive');
      for (var j=0; j<layers[i]; j++) {
        _neurons.add(Neuron(layer: i, number: j, activationFunction: function));
      }
      if (i != layers.length-1) {
        _neurons.add(Neuron(layer: i, number: layers[i], activationFunction: function, isCorrector: true));
      }
    }
    if (initialSynapses == null) {
      for (var i = 0; i < layers.length - 1; i++) {
        for (var j = 0; j <= layers[i]; j++) {
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

  List<double> process(List<double> input) {
    final stopwatch = Stopwatch();
    stopwatch.start();
    _neurons.forEach((n) => n.initNeuron());
    for (var i=0; i<_netConfiguration.first; i++) {
      _getNeuronCached(0, i).setExplicitValue(input[i]);
    }
    for (var i=0; i<_netConfiguration.length-1; i++) {
      _processSynapseLayer(i);
    }
    final res = <double>[];
    for (var i=0; i<_netConfiguration.last; i++) {
      res.add(_getNeuronCached(_netConfiguration.length-1, i).value);
    }
    stopwatch.stop();
    //print('processed in ${stopwatch.elapsedMilliseconds / 1000} sec');
    return res;
  }

  void train(List<TrainingData> trainData) {
    var counter = 0;
    for (var next in trainData) {
      if (next.inputData.length != _netConfiguration.first) {
        print('Data for training should have the same number of inputs as the neurons number of entry layer of the network, skipping');
        continue;
      }
      if (next.outputData.length != _netConfiguration.last) {
        print('Data for training should have the same number of outputs as the neurons number of exit layer of the network, skipping');
        continue;
      }
      process(next.inputData);
      for (var i=0; i<_netConfiguration.last; i++) {
        final neuron = _getNeuronCached(_netConfiguration.length-1, i);
        final sigma =
            (next.outputData[i] - neuron.value) * neuron.activationFunction.derivative(neuron.unsealedValue);
        for (var prevNeuron in neuron.prevLayerValues) {
          final deltaWeight = alpha * sigma * prevNeuron.neuronValue;
          _getSynapseCached(1, prevNeuron.neuronNumber, i).addWeight(deltaWeight);
          _getNeuronCached(1, prevNeuron.neuronNumber).addErrorValue(sigma * prevNeuron.synapseWeight);
        }
      }
      for (var i=0; i<_netConfiguration[1]; i++) {
        final neuron = _getNeuronCached(_netConfiguration.length-2, i);
        final sigma =
            neuron.error * neuron.activationFunction.derivative(neuron.unsealedValue);
        for (var prevNeuron in neuron.prevLayerValues) {
          final deltaWeight = alpha * sigma * prevNeuron.neuronValue;
          _getSynapseCached(0, prevNeuron.neuronNumber, i).addWeight(deltaWeight);
        }
      }
      counter++;
      if (counter % 100 == 0) {
        print('Trained cases: $counter');
      }
    }
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
    for (var i=0; i<=_netConfiguration[layer]; i++) {
      final origin = _getNeuronCached(layer, i);
      for (var j=0; j<_netConfiguration[layer+1]; j++) {
        final destination = _getNeuronCached(layer+1, j);
        final synapse = _getSynapseCached(layer, i, j);
        destination.addWeightedValue(
          TrainingNeuronDescription(neuronValue: origin.value, synapseWeight: synapse.weight, neuronNumber: i)
        );
      }
    }
    _neurons.where((n) => n.layer == layer + 1).forEach((f) => f.sealValue());
  }

  final _getNeuron = memo3((List<Neuron> neurons, int layer, int number) => neurons.singleWhere((n) => n.layer == layer && n.number == number));
  final _getSynapse = memo4((List<Synapse> synapses, int layer, int origin, int destination) =>
      synapses.singleWhere((s) => s.synapseLayer == layer && s.originNeuron == origin && s.destinationNeuron == destination));

  Neuron _getNeuronCached(int layer, int number) => _getNeuron(_neurons, layer, number);
  Synapse _getSynapseCached(int layer, int origin, int destination) => _getSynapse(_synapses, layer, origin, destination);

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
