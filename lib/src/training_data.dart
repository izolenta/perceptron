import 'dart:convert';

class TrainingData {
  final List<double> inputData;
  final List<double> outputData;

  TrainingData(this.inputData, this.outputData);

  String toJson() => jsonEncode({'i': inputData, 'o': outputData});
}