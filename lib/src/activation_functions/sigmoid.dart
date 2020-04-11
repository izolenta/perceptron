import 'dart:math';

import 'package:perceptron/src/activation_functions/activation_function.dart';

class Sigmoid extends ActivationFunction {
  @override
  double derivative(double param) => process(param) * (1 - process(param));

  @override
  double process(double param) => 1 / (1 + exp(-param));
}
