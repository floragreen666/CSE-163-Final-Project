"""
Flora Tang (1863826), Owen Wang (1831955)
CSE 163 AA

Calculates and plots graphs related to the median earnings of
former students working and not enrolled 10 years after entry.
"""


from util import (CollegeLoader, GeoLoader, CollegeDocumentLoader,
                  join_curdir, ensure_dir_exists)
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import seaborn as sns


def replace_legend_value_mappings(legends, documentation):
    """
    Takes a list of legend values, and a CollegeDocument object.

    Returns a list of corresponding labels for each legend value.
    """
    col_documentation = documentation.get_column(legends[0])
    dtype = col_documentation.column_type
    return [col_documentation.description] + \
        [col_documentation.values_mapping[dtype(l)] for l in legends[1:]]


def process_data_p1(data):
    """
    Takes a dataset.

    Returns a DataFrame contianing average median earnings of former students
    per college control type and academic year.
    """
    return data[["CONTROL", "Academic Year", "MD_EARN_WNE_P10"]] \
        .groupby(["CONTROL", "Academic Year"], as_index=False).mean()


def plot_p1(data, documentation):
    """
    Takes a dataset and a CollegeDocument object.

    Plots a titled line plot to visualize the yearly changes of the average
    median earnings of former students per college control type. The plot is
    colored by control types, with a legend indicating the meaning of colors.
    """
    fig, ax = plt.subplots(1)
    data = process_data_p1(data)
    sns.lineplot(x="Academic Year", y="MD_EARN_WNE_P10", hue="CONTROL",
                 data=data, ax=ax,
                 palette=sns.color_palette(["#9b59b6", "#3498db", "#e74c3c"]))
    fig.suptitle("Yearly Changes of Average Median Earnings of Former " +
                 "Students per College Control Type")
    ax.set(ylabel="Average Median Earnings ($)")
    handles, legends = ax.get_legend_handles_labels()
    legends = replace_legend_value_mappings(legends, documentation)
    ax.legend(handles, legends, loc="lower left")
    fig.savefig(join_curdir("results", "plot_p1.png"))


def draw_geo_bound(data, ax, percent=0.05, color="black", bound_width=1):
    """
    Takes a dataset, an axes, a size percent for the bound, a color for
    the bound, and a width for the bound.

    Plots the bound with the given parameters on the given axes with the given
    dataset.
    """
    x1, y1, x2, y2 = data.total_bounds
    w, h = x2 - x1, y2 - y1
    w_margin, h_margin = w * percent, h * percent
    rect = plt.Rectangle((x1 - w_margin, y1 - h_margin),
                         w + w_margin * 2, h + h_margin * 2,
                         color=color, fill=False,
                         lw=bound_width, clip_on=False)
    ax.add_patch(rect)


def plot_states(data, ax, column, title=None, draw_bound=True,
                bound_args=None, plot_args=None):
    """
    Takes a dataset, an axes, a column, an optional title, an optional flag of
    drawing bound, an optional dictionary of arguments for draw_geo_bound, and
    an optional dictionary of arguments for GeoDataFrame.plot.

    Plots all states in the given data on the given axes colored by the given
    column.
    """
    bound_args = bound_args if bound_args else {}
    plot_args = plot_args if plot_args else {}
    if draw_bound:
        draw_geo_bound(data, ax, **bound_args)
    if title:
        ax.set_title(title)
    data.plot(column=column, ax=ax, **plot_args)


def plot_p2_highlight_capital(data, ax, color, size=12):
    """
    Takes a data, an axes, a text color, and a font size.

    Draws a line which points to the capital with a label which shows state
    name and average median earnings of former students working and not
    enrolled 10 years after entry.
    """
    name = "District of Columbia"
    capital = data[data["name"] == name]
    c = capital.geometry.centroid.to_list()[0]
    cx, cy = c.x, c.y
    ax.plot([cx + 2.5, cx], [cy - 1.5, cy], color=color)
    val = int(capital["MD_EARN_WNE_P10"].iloc[0])
    label = "{n}\n{v: ^{len}}".format(n=name, v="$" + str(val), len=len(name))
    ax.text(cx + 3, cy - 3, label, color=color, fontsize=size)


def plot_p2_states(data, axs):
    """
    Takes a data and a tuple of axes.

    Plots the states in different axes which will share common min, max values.
    """
    vmin, vmax = (data["MD_EARN_WNE_P10"].min(),
                  data["MD_EARN_WNE_P10"].nlargest(2).iloc[1])
    plot_args = {"vmin": vmin, "vmax": vmax}
    plot_states(data[~data["name"].isin(["Alaska", "Hawaii"])], axs[0],
                "MD_EARN_WNE_P10", "Mainland", draw_bound=False,
                plot_args={"legend": True, "cax": axs[-1], **plot_args})
    plot_states(data[data["name"] == "Alaska"], axs[1], "MD_EARN_WNE_P10",
                "Alaska", plot_args=plot_args)
    plot_states(data[data["name"] == "Hawaii"], axs[2], "MD_EARN_WNE_P10",
                "Hawaii", plot_args=plot_args)


def process_data_p2(data, documentation, geodata):
    """
    Takes a college dataset, a geospatial dataset, and a college documentation.

    Combines the 2 datasets and returns the result as a GeoDataFrame with all
    state ids replaced by state names.
    """
    documentation_state = documentation.get_column("ST_FIPS")
    mean_per_state = data.groupby("ST_FIPS", as_index=False).mean()
    mean_per_state["ST_FIPS"] = mean_per_state["ST_FIPS"] \
        .map(documentation_state.values_mapping)
    return geodata.merge(mean_per_state, how="left",
                         left_on="name", right_on="ST_FIPS")


def plot_p2(data, geodata, documentation):
    """
    Takes a college dataset, a geospatial dataset, and a college documentation.

    Plots a titled map of United States colored by the average median earnings
    of former students per year per state, with a legend indicating the
    meaning of colors. For aesthetic and readability purposes, mainland,
    Alaska, and Hawaii is plotted as separate subplots in one figure.
    Highlights District of Columbia as an outlier.
    """
    d = process_data_p2(data, geodata, documentation)
    fig = plt.figure()
    fig.suptitle("Average Median Earnings of Former Students " +
                 "per Year in each State")
    gs = GridSpec(2, 3, width_ratios=[1, 1, 0.03], height_ratios=[8, 3],
                  wspace=0, hspace=0, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    axc = fig.add_subplot(gs[:, 2])
    [ax.set_axis_off() for ax in [ax1, ax2, ax3]]
    plot_p2_states(d, (ax1, ax2, ax3, axc))
    plot_p2_highlight_capital(d, ax1, "#483D8B")
    axc.set_ylabel("Average Median Earnings ($)")
    axc.yaxis.set_label_position("left")
    fig.savefig(join_curdir("results", "plot_p2.png"))


def process_data_p3(data):
    """
    Takes a dataset.

    Returns a DataFrame with new columns containing average median SAT/ACT
    score percentages per year.
    To calculate the median SAT score percentage for a college in one year,
    add up median SAT reading, math, and writing scores and divide by 2400
    (the full score of SAT before January 2016). To calculate the median ACT
    score percentage for a college in one year, divide ACT cumulative score by
    36 (the full score of ACT).
    """
    d = data[["MD_EARN_WNE_P10", "SATVRMID", "SATMTMID", "SATWRMID",
              "ACTCMMID"]].dropna()
    d["SAT_SCORE_%"] = (d["SATVRMID"] + d["SATMTMID"] + d["SATWRMID"]) / 2400
    d["ACT_SCORE_%"] = d["ACTCMMID"] / 36
    return d


def plot_p3(data):
    """
    Takes a dataset.

    Plots two quadratic regression lines in the same titled plot for all
    colleges’ average median SAT/ACT score percentages per year with respect
    to their average median earnings of former students per year. The plot is
    colored by test types, with a legend indicating the meaning of colors.
    """
    d = process_data_p3(data)
    fig, ax = plt.subplots(1)
    fig.suptitle("Average Median SAT/ACT Score Percentages per Year \nvs.\n" +
                 "Average Median Earnings of Former Students per Year")
    sns.regplot(x="SAT_SCORE_%", y="MD_EARN_WNE_P10", data=d, order=2, ax=ax,
                color="blue", scatter=False, label="SAT Score Percentage")
    sns.regplot(x="ACT_SCORE_%", y="MD_EARN_WNE_P10", data=d, order=2, ax=ax,
                color="red", scatter=False, label="ACT Score Percentage")
    ax.legend(loc="upper left")
    ax.set(ylabel="Average Median Earnings ($)", xlabel="Score Percentage")
    fig.savefig(join_curdir("results", "plot_p3.png"))


def main():
    sns.set()
    plt.rcParams['figure.figsize'] = (16.0, 8.0)
    document = CollegeDocumentLoader().load(join_curdir("data", "data.meta"))
    geodata = GeoLoader() \
        .load(join_curdir("data", "ne_110m_admin_1_states_provinces.shp"))
    data = CollegeLoader(document).load(join_curdir("data", "data.csv"))
    ensure_dir_exists(join_curdir("results"))
    plot_p1(data, document)
    plot_p2(data, document, geodata)
    plot_p3(data)


if __name__ == "__main__":
    main()
