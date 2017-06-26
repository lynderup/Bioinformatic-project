

def write_predictions(predictions, filename):
    with open(filename, "w") as file:

        for name, xs, zs, ps in predictions:
        # for name, length, xs, zs, ps in predictions:
        #
        #     xs = "".join([str(x) for x in xs])
        #     zs = "".join([str(z) for z in zs])
        #     ps = "".join([str(p) for p in ps])

            file.write(">%s \n" % name)
            file.write("%s \n" % xs)
            file.write("%s \n" % zs)
            file.write("%s \n\n" % ps)
