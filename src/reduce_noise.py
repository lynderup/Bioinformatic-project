import numpy as np

def count_in_window(window):
    number_of_is = 0
    number_of_Ms = 0
    number_of_os = 0


    for x in window:
        if x == "i":
            number_of_is += 1
        elif x == "M":
            number_of_Ms += 1
        elif x == "o":
            number_of_os += 1

    return number_of_is, number_of_Ms, number_of_os


def reduce_noise(predictions):

    noise_reduced_predictions = []
    window_radius = 2

    for name, xs, zs, ps in predictions:

        noise_reduced_ps = []

        for i, p in enumerate(ps):
            start = max(0, i - window_radius)
            end = min(len(ps) - 1, i + window_radius)

            j = np.argmax(count_in_window(ps[start:end]))

            new_p = ["i", "M", "o"][j]
            noise_reduced_ps.append(new_p)

        noise_reduced_predictions.append((name, xs, zs, "".join(noise_reduced_ps)))

    return noise_reduced_predictions



