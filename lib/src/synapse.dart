import 'package:meta/meta.dart';

class Synapse {
  final int synapseLayer;
  final int originNeuron;
  final int destinationNeuron;
  final double weight;

  Synapse({
    @required this.synapseLayer,
    @required this.originNeuron,
    @required this.destinationNeuron,
    this.weight = 0});

  Map<String, Object> toJson() {
    final result = <String, Object>{};
    result['layer'] = synapseLayer;
    result['origin'] = originNeuron;
    result['destination'] = destinationNeuron;
    result['weight'] = weight;
    return result;
  }
}