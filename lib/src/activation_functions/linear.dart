import 'package:perceptron/src/activation_functions/activation_function.dart';

class Linear extends ActivationFunction {
  @override
  double derivative(double param) => 0;

  @override
  double process(double param) => param;
}
