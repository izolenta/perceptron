class ActivationFunctionType {
  static const ActivationFunctionType linear = ActivationFunctionType._('linear');
  static const ActivationFunctionType sigmoid = ActivationFunctionType._('sigmoid');
  static const ActivationFunctionType bipolarSigmoid = ActivationFunctionType._('bipolarSigmoid');

  final String value;

  const ActivationFunctionType._(this.value);

  static final List<ActivationFunctionType> values = [
    linear,
    sigmoid,
    bipolarSigmoid
  ];

  static ActivationFunctionType parse(String value) => values.firstWhere((item) => item.value == value);

  @override
  String toString() => value;

}