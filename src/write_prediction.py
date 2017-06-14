


def write_predictions(predictions, filename):
    with open(filename, "w") as file:

        for name, xs, zs, ps in predictions:

            file.write(">test %s \n" % name)
            file.write("%s \n" % xs)
            file.write("%s \n" % zs)
            file.write("%s \n\n" % ps)
