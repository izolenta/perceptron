import 'package:meta/meta.dart';

class TrainingNeuronDescription {
  final double synapseWeight;
  final double neuronValue;
  final int neuronNumber;

  TrainingNeuronDescription({
    @required this.synapseWeight,
    @required this.neuronValue,
    @required this.neuronNumber,
  });
}