import os, sys, subprocess

energy = sys.argv[1]
input = sys.argv[2]
output = sys.argv[3]

if not os.path.exists(output):
    os.makedirs(output)

mttb = [
    "--set-key-value",
    "x1_label",
    "top pair mass",
    "--set-key-value",
    "x1_label_tex",
    "$m_{t\\bar{t}}$",
    "--set-key-value",
    "x1_unit",
    "GeV",
    "--set-key-value",
    "y_label",
    "differential x-sec",
    "--set-key-value",
    "y_label_tex",
    "$\\frac{d\\sigma}{dm_{t\\bar{t}}}$",
    "--set-key-value",
    "y_unit",
    "pb/GeV",
]

ptt = [
    "--set-key-value",
    "x1_label",
    "top transverse momentum",
    "--set-key-value",
    "x1_label_tex",
    "$pT_t$",
    "--set-key-value",
    "x1_unit",
    "GeV",
    "--set-key-value",
    "y_label",
    "differential x-sec",
    "--set-key-value",
    "y_label_tex",
    "$\\frac{d\\sigma}{dpT_t}$",
    "--set-key-value",
    "y_unit",
    "pb/GeV",
]

absyt = [
    "--set-key-value",
    "x1_label",
    "top abs rapidity",
    "--set-key-value",
    "x1_label_tex",
    "$|y_t|$",
    "--set-key-value",
    "x1_unit",
    "",
    "--set-key-value",
    "y_label",
    "differential x-sec",
    "--set-key-value",
    "y_label_tex",
    "$\\frac{d\\sigma}{d|y_t|}$",
    "--set-key-value",
    "y_unit",
    "pb",
]

yt = [
    "--set-key-value",
    "x1_label",
    "top rapidity",
    "--set-key-value",
    "x1_label_tex",
    "$y_t$",
    "--set-key-value",
    "x1_unit",
    "",
    "--set-key-value",
    "y_label",
    "differential x-sec",
    "--set-key-value",
    "y_label_tex",
    "$\\frac{d\\sigma}{dy_t}$",
    "--set-key-value",
    "y_unit",
    "pb",
]

absyttbar = [
    "--set-key-value",
    "x1_label",
    "top pair abs rapidity",
    "--set-key-value",
    "x1_label_tex",
    "$|y_{t\\bar{t}}|$",
    "--set-key-value",
    "x1_unit",
    "",
    "--set-key-value",
    "y_label",
    "differential x-sec",
    "--set-key-value",
    "y_label_tex",
    "$\\frac{d\\sigma}{d|y_{t\\bar{t}}|}$",
    "--set-key-value",
    "y_unit",
    "pb",
]

yttbar = [
    "--set-key-value",
    "x1_label",
    "top pair rapidity",
    "--set-key-value",
    "x1_label_tex",
    "$y_{t\\bar{t}}$",
    "--set-key-value",
    "x1_unit",
    "",
    "--set-key-value",
    "y_label",
    "differential x-sec",
    "--set-key-value",
    "y_label_tex",
    "$\\frac{d\\sigma}{dy_{t\\bar{t}}}$",
    "--set-key-value",
    "y_unit",
    "pb",
]

default = ["pineappl", "write", "--scale", "0.001"]

matrix_suffix = "_NNLO.QCD.pineappl.lz4"
new_suffix = ".pineappl.lz4"

gnf8, gnf13 = [], []


def integrate_grid(grid_path, type):
    if type == "1d":
        bins = subprocess.run(
            ["pineappl", "read", "-b", grid_path], capture_output=True, text=True
        )
        n_bins = len(bins.stdout.splitlines()) - 2
        bin_range = "0-" + str(n_bins - 1)
        output_grid_path = grid_path.replace(
            ".pineappl.lz4", "-INTEGRATED.pineappl.lz4"
        )
        subprocess.run(
            [
                "pineappl",
                "write",
                "--merge-bins",
                bin_range,
                "--remap",
                "0,1",
                grid_path,
                output_grid_path,
            ]
        )

    elif type == "2d":
        bins = subprocess.run(
            ["pineappl", "read", "-b", grid_path], capture_output=True, text=True
        )
        n_bins = len(bins.stdout.splitlines()) - 2
        bin_range = "0-" + str(n_bins - 1)
        remap_string = ""
        for i in range(n_bins + 1):
            remap_string += str(i) + ","
        remap_string = remap_string[:-1]
        intermediate_grid_path = grid_path.replace(
            ".pineappl.lz4", "-temp.pineappl.lz4"
        )
        output_grid_path = grid_path.replace(
            ".pineappl.lz4", "-INTEGRATED.pineappl.lz4"
        )
        subprocess.run(
            [
                "pineappl",
                "write",
                "--remap",
                remap_string,
                grid_path,
                intermediate_grid_path,
            ]
        )
        subprocess.run(
            [
                "pineappl",
                "write",
                "--merge-bins",
                bin_range,
                "--remap",
                "0,1",
                intermediate_grid_path,
                output_grid_path,
            ]
        )
        subprocess.run(["rm", intermediate_grid_path])
    else:
        print("Invalid type specified. Please use '1d' or '2d'.")


def process_8():

    ################# 1D distributions

    # ATLAS_TTBAR_8TEV_2L_DIF_MTTBAR
    print("processing ATLAS_TTBAR_8TEV_2L_DIF_MTTBAR")
    input_path = input + "/ATLAS_TTBAR_8TEV_2L_DIF_MTTBAR" + matrix_suffix
    output_path = output + "/ATLAS_TTBAR_8TEV_2L_DIF_MTTBAR" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + mttb + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf8.append(input_path)

    # ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR
    print("processing ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR")
    input_path = input + "/ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR" + matrix_suffix
    output_path = output + "/ATLAS_TTBAR_8TEV_2L_DIF_YTTBAR" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + absyttbar + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf8.append(input_path)

    # ATLAS_TTBAR_8TEV_LJ_DIF_MTTBAR
    print("processing ATLAS_TTBAR_8TEV_LJ_DIF_MTTBAR")
    input_path = input + "/ATLAS_TTBAR_8TEV_LJ_DIF_MTTBAR" + matrix_suffix
    output_path = output + "/ATLAS_TTBAR_8TEV_LJ_DIF_MTTBAR" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + mttb + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf8.append(input_path)

    # ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR
    print("processing ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR")
    input_path = input + "/ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR" + matrix_suffix
    output_path = output + "/ATLAS_TTBAR_8TEV_LJ_DIF_YTTBAR" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + absyttbar + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf8.append(input_path)

    # ATLAS_TTBAR_8TEV_LJ_DIF_YT
    print("processing ATLAS_TTBAR_8TEV_LJ_DIF_YT")
    input_path = input + "/ATLAS_TTBAR_8TEV_LJ_DIF_YT" + matrix_suffix
    output_path = output + "/ATLAS_TTBAR_8TEV_LJ_DIF_YT" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + absyt + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf8.append(input_path)

    # ATLAS_TTBAR_8TEV_LJ_DIF_PTT
    print("processing ATLAS_TTBAR_8TEV_LJ_DIF_PTT")
    input_path = input + "/ATLAS_TTBAR_8TEV_LJ_DIF_PTT" + matrix_suffix
    output_path = output + "/ATLAS_TTBAR_8TEV_LJ_DIF_PTT" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + ptt + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf8.append(input_path)

    # CMS_TTBAR_8TEV_LJ_DIF_MTTBAR
    print("processing CMS_TTBAR_8TEV_LJ_DIF_MTTBAR")
    input_path = input + "/CMS_TTBAR_8TEV_LJ_DIF_MTTBAR" + matrix_suffix
    output_path = output + "/CMS_TTBAR_8TEV_LJ_DIF_MTTBAR" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + mttb + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf8.append(input_path)

    # CMS_TTBAR_8TEV_LJ_DIF_YTTBAR (catr)
    print("processing CMS_TTBAR_8TEV_LJ_DIF_YTTBAR")
    input_path = input + "/CMS_TTBAR_8TEV_LJ_DIF_YTTBAR_catr" + matrix_suffix

    intermediate_path = output + "/CMS_TTBAR_8TEV_LJ_DIF_YTTBAR_intermediate"
    output_path = output + "/CMS_TTBAR_8TEV_LJ_DIF_YTTBAR" + new_suffix
    if os.path.exists(input_path):

        subprocess.run(default + yttbar + [input_path, output_path])
        integrate_grid(output_path, "1d")
        subprocess.run(["rm", output_path])

        subprocess.run(
            ["python", "./matrix_yttbar.py", "-g", input_path, "-o", intermediate_path]
        )
        subprocess.run(default + yttbar + [intermediate_path + new_suffix, output_path])
        subprocess.run(["rm", intermediate_path + new_suffix])

    else:
        gnf8.append(input_path)

    # CMS_TTBAR_8TEV_LJ_DIF_YT
    print("processing CMS_TTBAR_8TEV_LJ_DIF_YT")
    input_path = input + "/CMS_TTBAR_8TEV_LJ_DIF_YT" + matrix_suffix
    output_path = output + "/CMS_TTBAR_8TEV_LJ_DIF_YT" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + yt + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf8.append(input_path)

    # CMS_TTBAR_8TEV_LJ_DIF_PTT
    print("processing CMS_TTBAR_8TEV_LJ_DIF_PTT")
    input_path = input + "/CMS_TTBAR_8TEV_LJ_DIF_PTT" + matrix_suffix
    output_path = output + "/CMS_TTBAR_8TEV_LJ_DIF_PTT" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + ptt + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf8.append(input_path)

    ################# 2D distributions

    # CMS_TTBAR_8TEV_2L_DIF_PTT-YT
    print("processing CMS_TTBAR_8TEV_2L_DIF_PTT-YT")
    input_path = input + "/CMS_TTBAR_8TEV_2L_DIF_PTT-YT" + matrix_suffix
    output_path = output + "/CMS_TTBAR_8TEV_2L_DIF_PTT-YT" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(
            default
            + [
                "--set-key-value",
                "x1_label",
                "top abs rapidity",
                "--set-key-value",
                "x1_label_tex",
                "$|y_t|$",
                "--set-key-value",
                "x1_unit",
                "",
            ]
            + [
                "--set-key-value",
                "x2_label",
                "top transverse momentum",
                "--set-key-value",
                "x2_label_tex",
                "$pT_t$",
                "--set-key-value",
                "x2_unit",
                "GeV",
            ]
            + [
                "--set-key-value",
                "y_label",
                "double diff x-sec",
                "--set-key-value",
                "y_label_tex",
                "$\\frac{d^2\\sigma}{d|y_t|dpT_t}$",
                "--set-key-value",
                "y_unit",
                "pb/GeV",
            ]
            + [input_path, output_path]
        )
        integrate_grid(output_path, "2d")
    else:
        gnf8.append(input_path)

    # CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT
    print("processing CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT")
    input_path = input + "/CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT" + matrix_suffix
    output_path = output + "/CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YT" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(
            default
            + [
                "--set-key-value",
                "x1_label",
                "top pair mass",
                "--set-key-value",
                "x1_label_tex",
                "$m_{t\\bar{t}}$",
                "--set-key-value",
                "x1_unit",
                "GeV",
            ]
            + [
                "--set-key-value",
                "x2_label",
                "top abs rapidity",
                "--set-key-value",
                "x2_label_tex",
                "$|y_t|$",
                "--set-key-value",
                "x2_unit",
                "",
            ]
            + [
                "--set-key-value",
                "y_label",
                "double diff x-sec",
                "--set-key-value",
                "y_label_tex",
                "$\\frac{d^2\\sigma}{dm_{t\\bar{t}}d|y_t|}$",
                "--set-key-value",
                "y_unit",
                "pb/GeV",
            ]
            + [input_path, output_path]
        )
        integrate_grid(output_path, "2d")
    else:
        gnf8.append(input_path)

    # CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YTTBAR
    print("processing CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YTTBAR")
    input_path = input + "/CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YTTBAR" + matrix_suffix
    output_path = output + "/CMS_TTBAR_8TEV_2L_DIF_MTTBAR-YTTBAR" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(
            default
            + [
                "--set-key-value",
                "x1_label",
                "top pair mass",
                "--set-key-value",
                "x1_label_tex",
                "$m_{t\\bar{t}}$",
                "--set-key-value",
                "x1_unit",
                "GeV",
            ]
            + [
                "--set-key-value",
                "x2_label",
                "top pair abs rapidity",
                "--set-key-value",
                "x2_label_tex",
                "$|y_{t\\bar{t}}|$",
                "--set-key-value",
                "x2_unit",
                "",
            ]
            + [
                "--set-key-value",
                "y_label",
                "double diff x-sec",
                "--set-key-value",
                "y_label_tex",
                "$\\frac{d^2\\sigma}{dm_{t\\bar{t}}d|y_{t\\bar{t}}|}$",
                "--set-key-value",
                "y_unit",
                "pb",
            ]
            + [input_path, output_path]
        )
        integrate_grid(output_path, "2d")
    else:
        gnf8.append(input_path)

    if len(gnf8) == 0:
        print("All grids processed successfully")
    else:
        print("The following grids were not found:")
        for i in gnf8:
            print(i)
        print("Please check the input path and try again.")


def process_13():

    ################# 1D distributions

    # ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR
    print("processing ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR")
    input_path = input + "/ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR" + matrix_suffix
    output_path = output + "/ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + mttb + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf13.append(input_path)

    # ATLAS_TTBAR_13TEV_LJ_DIF_PTT
    print("processing ATLAS_TTBAR_13TEV_LJ_DIF_PTT")
    input_path = input + "/ATLAS_TTBAR_13TEV_LJ_DIF_PTT" + matrix_suffix
    output_path = output + "/ATLAS_TTBAR_13TEV_LJ_DIF_PTT" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + ptt + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf13.append(input_path)

    # ATLAS_TTBAR_13TEV_LJ_DIF_YT
    print("processing ATLAS_TTBAR_13TEV_LJ_DIF_YT")
    input_path = input + "/ATLAS_TTBAR_13TEV_LJ_DIF_YT" + matrix_suffix
    output_path = output + "/ATLAS_TTBAR_13TEV_LJ_DIF_YT" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + yt + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf13.append(input_path)

    # ATLAS_TTBAR_13TEV_LJ_DIF_YTTBAR
    print("processing ATLAS_TTBAR_13TEV_LJ_DIF_YTTBAR")
    input_path = input + "/ATLAS_TTBAR_13TEV_LJ_DIF_YTTBAR" + matrix_suffix
    output_path = output + "/ATLAS_TTBAR_13TEV_LJ_DIF_YTTBAR" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + yttbar + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf13.append(input_path)

    # ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR
    print("processing ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR")
    input_path = input + "/ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR" + matrix_suffix
    output_path = output + "/ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + mttb + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf13.append(input_path)

    # ATLAS_TTBAR_13TEV_HADR_DIF_YTTBAR
    print("processing ATLAS_TTBAR_13TEV_HADR_DIF_YTTBAR")
    input_path = input + "/ATLAS_TTBAR_13TEV_HADR_DIF_YTTBAR" + matrix_suffix
    output_path = output + "/ATLAS_TTBAR_13TEV_HADR_DIF_YTTBAR" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + yttbar + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf13.append(input_path)

    # CMS_TTBAR_13TEV_2L_DIF_MTTBAR
    print("processing CMS_TTBAR_13TEV_2L_DIF_MTTBAR")
    input_path = input + "/CMS_TTBAR_13TEV_2L_DIF_MTTBAR" + matrix_suffix
    output_path = output + "/CMS_TTBAR_13TEV_2L_DIF_MTTBAR" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + mttb + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf13.append(input_path)

    # CMS_TTBAR_13TEV_2L_DIF_YTTBAR (catr)
    print("processing CMS_TTBAR_13TEV_2L_DIF_YTTBAR")
    input_path = input + "/CMS_TTBAR_13TEV_2L_DIF_YTTBAR_catr" + matrix_suffix
    intermediate_path = output + "/CMS_TTBAR_13TEV_2L_DIF_YTTBAR_intermediate"
    output_path = output + "/CMS_TTBAR_13TEV_2L_DIF_YTTBAR" + new_suffix
    if os.path.exists(input_path):

        subprocess.run(default + yttbar + [input_path, output_path])
        integrate_grid(output_path, "1d")
        subprocess.run(["rm", output_path])

        subprocess.run(
            ["python", "./matrix_yttbar.py", "-g", input_path, "-o", intermediate_path]
        )
        subprocess.run(default + yttbar + [intermediate_path + new_suffix, output_path])
        subprocess.run(["rm", intermediate_path + new_suffix])

    else:
        gnf13.append(input_path)

    # CMS_TTBAR_13TEV_2L_DIF_YT
    print("processing CMS_TTBAR_13TEV_2L_DIF_YT")
    input_path = input + "/CMS_TTBAR_13TEV_2L_DIF_YT" + matrix_suffix
    output_path = output + "/CMS_TTBAR_13TEV_2L_DIF_YT" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + yt + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf13.append(input_path)

    # CMS_TTBAR_13TEV_2L_DIF_PTT
    print("processing CMS_TTBAR_13TEV_2L_DIF_PTT")
    input_path = input + "/CMS_TTBAR_13TEV_2L_DIF_PTT" + matrix_suffix
    output_path = output + "/CMS_TTBAR_13TEV_2L_DIF_PTT" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + ptt + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf13.append(input_path)

    # CMS_TTBAR_13TEV_LJ_DIF_MTTBAR
    print("processing CMS_TTBAR_13TEV_LJ_DIF_MTTBAR")
    input_path = input + "/CMS_TTBAR_13TEV_LJ_DIF_MTTBAR" + matrix_suffix
    output_path = output + "/CMS_TTBAR_13TEV_LJ_DIF_MTTBAR" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + mttb + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf13.append(input_path)

    # CMS_TTBAR_13TEV_LJ_DIF_YTTBAR
    print("processing CMS_TTBAR_13TEV_LJ_DIF_YTTBAR")
    input_path = input + "/CMS_TTBAR_13TEV_LJ_DIF_YTTBAR" + matrix_suffix
    output_path = output + "/CMS_TTBAR_13TEV_LJ_DIF_YTTBAR" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + yttbar + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf13.append(input_path)

    # CMS_TTBAR_13TEV_LJ_DIF_YT
    print("processing CMS_TTBAR_13TEV_LJ_DIF_YT")
    input_path = input + "/CMS_TTBAR_13TEV_LJ_DIF_YT" + matrix_suffix
    output_path = output + "/CMS_TTBAR_13TEV_LJ_DIF_YT" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + yt + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf13.append(input_path)

    # CMS_TTBAR_13TEV_LJ_DIF_PTT
    print("processing CMS_TTBAR_13TEV_LJ_DIF_PTT")
    input_path = input + "/CMS_TTBAR_13TEV_LJ_DIF_PTT" + matrix_suffix
    output_path = output + "/CMS_TTBAR_13TEV_LJ_DIF_PTT" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(default + ptt + [input_path, output_path])
        integrate_grid(output_path, "1d")
    else:
        gnf13.append(input_path)

    ################# 2D distributions

    # ATLAS_TTBAR_13TEV_LJ_DIF_PTT-YT
    print("processing ATLAS_TTBAR_13TEV_LJ_DIF_PTT-YT")
    input_path = input + "/ATLAS_TTBAR_13TEV_LJ_DIF_PTT-YT_bm" + matrix_suffix
    output_path = output + "/ATLAS_TTBAR_13TEV_LJ_DIF_PTT-YT" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(
            default
            + [
                "--set-key-value",
                "x1_label",
                "top abs rapidity",
                "--set-key-value",
                "x1_label_tex",
                "$|y_t|$",
                "--set-key-value",
                "x1_unit",
                "",
            ]
            + [
                "--set-key-value",
                "x2_label",
                "top transverse momentum",
                "--set-key-value",
                "x2_label_tex",
                "$pT_t$",
                "--set-key-value",
                "x2_unit",
                "GeV",
            ]
            + [
                "--set-key-value",
                "y_label",
                "double diff x-sec",
                "--set-key-value",
                "y_label_tex",
                "$\\frac{d^2\\sigma}{d|y_t|dpT_t}$",
                "--set-key-value",
                "y_unit",
                "pb/GeV",
            ]
            + ["--merge-bins", "3-4,9-11,14-15,16-17"]
            + [input_path, output_path]
        )
        integrate_grid(output_path, "2d")
    else:
        gnf13.append(input_path)

    # ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR-PTT
    print("processing ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR-PTT")
    input_path = input + "/ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR-PTT_bm" + matrix_suffix
    output_path = output + "/ATLAS_TTBAR_13TEV_LJ_DIF_MTTBAR-PTT" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(
            default
            + [
                "--set-key-value",
                "x1_label",
                "top pair mass",
                "--set-key-value",
                "x1_label_tex",
                "$m_{t\\bar{t}}$",
                "--set-key-value",
                "x1_unit",
                "GeV",
            ]
            + [
                "--set-key-value",
                "x2_label",
                "top transverse momentum",
                "--set-key-value",
                "x2_label_tex",
                "$pT_t$",
                "--set-key-value",
                "x2_unit",
                "GeV",
            ]
            + [
                "--set-key-value",
                "y_label",
                "double diff x-sec",
                "--set-key-value",
                "y_label_tex",
                "$\\frac{d^2\\sigma}{dm_{t\\bar{t}}dpT_t}$",
                "--set-key-value",
                "y_unit",
                "$pb/GeV^2$",
            ]
            + [
                "--merge-bins",
                "0-1,2-3,4-7,9-10,11-13,14-15,17-18,19-20,21-22,24-27,28-29,30-31",
            ]
            + [input_path, output_path]
        )
        integrate_grid(output_path, "2d")
    else:
        gnf13.append(input_path)

    # ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR
    print("processing ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR")
    input_path1 = (
        input + "/ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR_gm1" + matrix_suffix
    )
    input_path2 = (
        input + "/ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR_gm2" + matrix_suffix
    )
    input_path3 = (
        input + "/ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR_gm3" + matrix_suffix
    )
    temp_path = output + "/ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR_temp" + new_suffix
    output_path = output + "/ATLAS_TTBAR_13TEV_HADR_DIF_MTTBAR-YTTBAR" + new_suffix
    if os.path.exists(input_path1):
        subprocess.run(
            ["pineappl", "merge"] + [temp_path, input_path1, input_path2, input_path3]
        )
        subprocess.run(
            default
            + [
                "--set-key-value",
                "x1_label",
                "top pair mass",
                "--set-key-value",
                "x1_label_tex",
                "$m_{t\\bar{t}}$",
                "--set-key-value",
                "x1_unit",
                "GeV",
            ]
            + [
                "--set-key-value",
                "x2_label",
                "top pair abs rapidity",
                "--set-key-value",
                "x2_label_tex",
                "$|y_{t\\bar{t}}|$",
                "--set-key-value",
                "x2_unit",
                "",
            ]
            + [
                "--set-key-value",
                "y_label",
                "double diff x-sec",
                "--set-key-value",
                "y_label_tex",
                "$\\frac{d^2\\sigma}{dm_{t\\bar{t}}d|y_{t\\bar{t}}|}$",
                "--set-key-value",
                "y_unit",
                "$pb/GeV$",
            ]
            + [temp_path, output_path]
        )
        integrate_grid(output_path, "2d")
        subprocess.run(["rm", temp_path])
    else:
        gnf13.append(input_path1)
        gnf13.append(input_path2)
        gnf13.append(input_path3)

    # CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-YTTBAR
    print("processing CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-YTTBAR")
    input_path = input + "/CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-YTTBAR" + matrix_suffix
    output_path = output + "/CMS_TTBAR_13TEV_LJ_DIF_MTTBAR-YTTBAR" + new_suffix
    if os.path.exists(input_path):
        subprocess.run(
            default
            + [
                "--set-key-value",
                "x1_label",
                "top pair mass",
                "--set-key-value",
                "x1_label_tex",
                "$m_{t\\bar{t}}$",
                "--set-key-value",
                "x1_unit",
                "GeV",
            ]
            + [
                "--set-key-value",
                "x2_label",
                "top pair abs rapidity",
                "--set-key-value",
                "x2_label_tex",
                "$|y_{t\\bar{t}}|$",
                "--set-key-value",
                "x2_unit",
                "",
            ]
            + [
                "--set-key-value",
                "y_label",
                "double diff x-sec",
                "--set-key-value",
                "y_label_tex",
                "$\\frac{d^2\\sigma}{dm_{t\\bar{t}}d|y_{t\\bar{t}}|}$",
                "--set-key-value",
                "y_unit",
                "$pb/GeV$",
            ]
            + [input_path, output_path]
        )
        integrate_grid(output_path, "2d")
    else:
        gnf13.append(input_path)

    if len(gnf13) == 0:
        print("All grids processed successfully")
    else:
        print("The following grids were not found:")
        for i in gnf13:
            print(i)
        print("Please check the input path and try again.")


if __name__ == "__main__":
    if energy == "8":
        process_8()
    elif energy == "13":
        process_13()
    else:
        print("accepted arguments are 8 or 13")
