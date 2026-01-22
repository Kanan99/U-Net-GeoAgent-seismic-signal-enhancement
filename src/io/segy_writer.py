import segyio

def write_segy(template_path, output_path, data):
    with segyio.open(template_path, "r") as src:
        spec = segyio.spec()
        spec.samples = src.samples
        spec.format = src.format
        spec.tracecount = data.shape[1]

        with segyio.create(output_path, spec) as dst:
            for i in range(data.shape[1]):
                dst.trace[i] = data[:, i]
