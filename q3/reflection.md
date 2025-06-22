## Reflection: Perceptron From Scratch

Initially, the perceptron produced nearly random outputs due to randomly initialized weights. Accuracy was low and loss was high. After several hundred epochs of gradient descent, the model successfully classified apples and bananas with over 95% accuracy. This highlights how even a single-layer model can learn meaningful patterns with the right optimization.

The learning rate (0.1) had a significant impact on training. A smaller rate like 0.01 led to slow convergence, while a larger one like 0.5 caused oscillations and unstable loss. The chosen rate struck a good balance between speed and stability, reaching <0.05 loss within 600 epochs.

A helpful analogy is the "DJ knob" or "child learning" metaphor: Just like a child gradually adjusts their behavior after feedback, the perceptron updates weights incrementally to get better at a task. The learning rate is like how sensitive the child is to correction — too sensitive (high LR), and learning becomes erratic; too insensitive, and progress is slow.

This project provided a great hands-on understanding of how logistic regression works under the hood.
