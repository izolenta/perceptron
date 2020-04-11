import 'package:meta/meta.dart';

class Synapse {
  final int synapseLayer;
  final int originNeuron;
  final int destinationNeuron;

  double _weight;
  double get weight => _weight;

  Synapse({
    @required this.synapseLayer,
    @required this.originNeuron,
    @required this.destinationNeuron,
    double weight = 0}) {
    _weight = weight;
  }

  void addWeight(double delta) {
    _weight += delta;
  }

  Map<String, Object> toJson() {
    final result = <String, Object>{};
    result['layer'] = synapseLayer;
    result['origin'] = originNeuron;
    result['destination'] = destinationNeuron;
    result['weight'] = weight;
    return result;
  }
}