import 'dart:math';

import 'package:perceptron/src/activation_functions/activation_function.dart';

class BipolarSigmoid extends ActivationFunction {
  @override
  double derivative(double param) => 0.5 * (1 + process(param)) * (1 - process(param)) ;

  @override
  double process(double param) =>  2 / (1 + exp(-param)) - 1;
}
