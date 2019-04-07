# COURSE 7: DATA VISUALIZATIO WITH PYTHON

**Syllabus:**

Tools: `matplotlib`, `seaborn`, and `folium`

	Week 1 - Introduction to Data Visualization Tools
		Introduction to Data Visualization
		Introduction to Matplotlib
		Basic Plotting with Matplotlib
		Dataset on Immigration to Canada
		Line Plots
		Lab: Introduction to Matplotlib and Line Plots
		Quiz: Introduction to Data Visualization Tools
	Week 2 - Basic and Specialized Visualization Tools
		Area Plots
		Histograms
		Bar Charts
		Pie Charts
		Box Plots
		Scatter Plots
		Bubble Plots
		Lab: Basic Visualization Tools
		Lab: Specialized Visualization Tools
		Quiz: Basic Visualization Tools
		Quiz: Specialized Visualization Tools
	Week 3 - Advanced Visualizations and Geospatial Data
		Waffle Charts
		Word Clouds
		Seaborn and Regression Plots
		Introduction to Folium and Map Styles
		Maps with Markers
		Choropleth Maps
		Lab: Advanced Visualization Tools
		Lab: Creating Maps and Visualizing Geospatial Data
		Quiz: Advanced Visualization Tools
		Quiz: Visualizing Geospatial Data
		Peer-review Assignment

## Week 1 - Introduction to Data Visualization Tools


### 1.1. Introduction to Data Visualization

Why Build Visuals?
1. For exploratory data analysis
2. Communicate data clearly
3. Share unbiased representation of data
4. Use them to support recommendations to different stakeholders

Best Practices: When creating a visual, always remember:
1. Less is more *effective*
2. Less is more *attractive*
3. Less is more *impactive*

Reference: [Dark Horse Analytics Co.](http://darkhorseanalytics.com/)

Example:

An ugly pie chart figure:

![c7_fig_ugly_pig_meat_piechart](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c7_fig_ugly_pig_meat_piechart.png)

Now apply the recommendations of Dark Horse Analytics: Remove background, borders, redundant legends, 3D, text bolding, and reduce color

![c7_fig_ugly_pig_meat_piechart_2](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c7_fig_ugly_pig_meat_piechart_2.png)

Another way to illustrate the result:

![c7_fig_ugly_pig_meat_piechart_3](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c7_fig_ugly_pig_meat_piechart_3.png)



### 1.2. Introduction to Matplotlib

History:
* Created by John Hunter, who was a neurobiologist and was part of a research team that was working on analyzing Electrocorticography (ECoG) signals

Matplotlib's architecture is composed of three main layers: 

1. `Back-end` layer: has three built-in abstract interface classes: `FigureCanvas`, `Renderer`, `Event`
	* `FigureCanvas` (`matplotlib.backend_based.FigureCanvas`): defines and encompasses the area on which the figure is drawn
	* `Renderer` (`matplotlib.backend_based.Renderer`): an instance of the renderer class knows how to draw on the figure canvas
	* `Event` (`matplotlib.backend_based.Event`): handles user inputs such as keyboard strokes and mouse clicks

2. `Artist` layer: 
	* Comprised of one main object---the `Artist`. The `Artist` is the object that knows how to take the `Renderer` and use it to put ink on the canvas
	* Everything you see on a `Matplotlib` figure is an `Artist` instance
	* Titles, lines, tick labels, images, and so on, all correspond to individual `Artist` instances 
	* Where much of the heavy lifting happens and is usually the appropriate programming paradigm when writing a web application server, or a UI application, or perhaps a script to be shared with other developers
	* Two types of `Artist` objects:
		* Primitive type: Line2D, Rectangle, Circle, or Text
		* Composite type: Axis, Tick, Axes, and Figure
	* The top-level `Matplotlib` object that contains and manages all of the elements in a given graphic is the `Figure` artist, and the most important composite artist is the `Axes` because it is where most of the `Matplotlib API` plotting methods are defined, including methods to create and manipulate the ticks, the axis lines, the grid or the plot background
	* Each **composite** artist may contain other **composite** artists as well as **primitive** artists. So a figure artist for example would contain an axis artist as well as a rectangle or text artists.

3. `Scripting` layer: 
	* Comprised mainly of `pylot`, a scripting interface that is lighter than the `Artist` layer
	* Appropriate layer for everyday purposes and is considered a lighter scripting interface to simplify common tasks and for a quick and easy generation of graphics and plots. 

**Example:**

```python
from matplotlib.backend_based_agg import FigureCanvasAgg as FigureCanvas # import FigureCanvas
from matplotlib.figure import Figure 	# import Figure artist
import matplotlib.pylot as plt 			# scripting layer

fig = Figure()
canvas = FigureCanvas(fig)

# Create 10000 random numbers using numpy
import numpy as np 

x = np.random.randn(10000)
ax = fig.add_subplot(111) # create an axes artist

ax.hist(x, 100) # generate a histogram of the 10000 numbers
ax.set_title('Normal distribution with $\mu=0, \sigma=1$')
fig.savefig('matplotlib_histogram.png')

# Using scripting layer (pylot - plt):
plt.hist(x, 100) # generate a histogram of the 10000 numbers
plt.title(r'Normal distribution with $\mu=0, \sigma=1$')
plt.savefig('matplotlib_histogram.png')
plt.show()

```

![c7_fig_matplotlib_example](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c7_fig_matplotlib_example.png)

Further reading: http://www.aosabook.org/en/matplotlib.html


### 1.3. Basic Plotting with Matplotlib

* Using Jupyter notebook: Magic function: `%matplotlib`, e.g., `%matplotlib inline` to enforce plots to be rendered within the browser (Jupyter notebook), and pass in `inline` as the backend.
* Matplotlib has a number of different backends available. One limitation of this backend is that *you cannot modify a figure once it is rendered*. So after rendering the above figure, there is no way for us to add, for example, a figure title or label its axes. You will need to generate a new plot and add a title and the axes labels before calling the show function. 
* A backend that overcomes this limitation is the `notebook backend`
	* With the notebook backend in place, if a plt function is called, it checks if an active figure exists, and any functions you call will be applied to this active figure
	* If a figure does not exist, it renders a new figure. So when we call the plt.plot function to plot a circular mark at position `(5, 5)`, the backend checks if an active figure exists since there isn't an active figure it generates a figure and adds a circular mark to position (5, 5)	
	* And what is beautiful about this back end is that now we can easily add a title for example or labels to the axes after the plot was rendered, without the need to regenerate the figure
* `pandas` also has a built-in implementation of plotting

Example of plotting using `pandas`:
```python
# Given pandas dataframe df
df.plot(kind = "line")	# line plot
df.plot(kind = "hist")	# histogram plot
```

![c7_fig_pandas_plot_example](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c7_fig_pandas_plot_example.png)

![c7_fig_pandas_plot_example2](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c7_fig_pandas_plot_example2.png)


### 1.4. Dataset on Immigration to Canada

```python
import numpy as np
import pandas as pd 
from __future__ import print_function # adds compatibility to python2

# install xlrd
!pip install xlrd

print('xlrd installed!')

df_can = pd.read_excel(
	'https://ibm.box.com/shared/static/....xlsx',
	sheetname = "Canada by Citizenship",
	skiprows = range(20), 	# skip first 20 rows
	skip_footer = 2)

# Display the dataframe
df_can.head()	# 5 first rows

```

### 1.5. Line Plots

```python
import matplotlib as mpl 
import matplotlib.pylot as plt 

years = list(map(str, range(1980, 2014)))
df_can.loc['Haiti', years].plot(kind = 'line') # plot using pandas function
plt.title('Immigration from Haiti')
plt.ylabel('Number of immigrants')
plt.xlable('Years')
plt.show()
```

![c7_fig_can_immigration_plot](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c7_fig_can_immigration_plot.png)

### LAB 1: Introduction to Matplotlib and Line Plots
### QUIZ 1: Introduction to Data Visualization Tools



## Week 2 - Basic and Specialized Visualization Tools

### 2.1. Area Plots
* Also known as area chart or area graph
* Commonly used to represent cumulated totals using numbers or percentages over time
* Is based on the line plot

Generating Area Plots
```python
df_canada.sort_values(['Total'], ascending=False, axis=0, inplace=True)
```

NOTE: `Matplotlib` plots the indices of a dataframe on the horizontal axis, and with the dataframe as shown, the countries will be plotted on the horizontal axis. So to fix this, we need to take the transpose of the dataframe

```python
years = list(map(str, range(1980, 2014)))
df_canada.sort_values(['Total'], ascending=False, axis=0, inplace=True)
df_top5 = df_canada.head()
df_top5 = df_top5[years].transpose()

# Now plot
df_top5.plot(kind='area')
plt.title('Immigration trend of top 5 countries')
plt.ylabel('Number of immigrants')
plt.xlable('Years')
plt.show()
```

![c7_w2_immigration_trend](https://github.com/luuquangtrung/ibm_data_science/blob/master/images/c7_w2_immigration_trend.png)


### 2.2. Histograms
* Is a way of representing the frequency distribution of a variable



### 2.3. Bar Charts

### 2.4. Pie Charts
* A pie chart is a circular statistical graphic divided into slices to illustrate numerical proportion
* Most argue that pie charts fail to accurately display data with any consistency. Bar charts are much better when it comes to representing the data in a consistent way and getting the message across

### 2.5. Box Plots


### 2.6. Scatter Plots


### 2.7. Bubble Plots








### LAB 1: Basic Visualization Tools
### LAB 2: Specialized Visualization Tools
### QUIZ 1: Basic Visualization Tools
### QUIZ 2 Specialized Visualization Tools

## Week 3 - Advanced Visualizations and Geospatial Data
### 3.1. Waffle Charts
*  A waffle chart is a great way to visualize data in relation to a whole or to highlight progress against a given threshold
* For example, say immigration from Scandinavia to Canada is comprised only of immigration from Denmark, Norway, and Sweden, and we're interested in visualizing the contribution of each of these countries to the Scandinavian immigration to Canada. The main idea here is for a given waffle chart whose desired height and width are defined, the contribution of each country is transformed into a number of tiles that is proportional to the country's contribution to the total, so that more the contribution the more the tiles, resulting in what resembles a waffle when combined
* Unfortunately Matplotlib does not have a built-in function to create waffle charts. 



### 3.2. Word Clouds
### 3.3. Seaborn and Regression Plots
### 3.4. Introduction to Folium and Map Styles

Folium:
* A powerful data visualization library in Python that was built primarily to help people visualize geospatial data.
* With Folium, you can create a map of any location in the world as long as you know its latitude and longitude values
* Folium enables both the binding of data to a map for choropleth (i.e., thematic map) visualizations as well as passing visualization as markers on the map
* The library has a number of built-in tilesets from `OpenStreetMap`, `Mapbox`, and `Stamen`, and support custom tilesets with `Mapbox API` keys
* The map is **interactive**, i.e., one can zoom in and zoom out after the map is rendered
* The default map style is the `Open Street Map`, which shows a street view of an area when you're zoomed in and shows the borders of the world countries when you're zoomed all the way out

```python
# Define the world map
world_map = folium.Map()

# Display world map
world_map

# Map of Canada
canada_map = folium.Maps(
	location=[56.130, -106.35], # center
	zoom_start=4,				# zoom level
	tiles='Stamen Toner' 		# map style
)

# Styles: Stamen Toner, Stamen Terrain
```

**NOTE:** 
* `Stamen Toner`: great for visualizing and exploring river meanders and coastal zones
* `Stamen Terrain`: great for visualizing hill shading and natural vegetation colors

### 3.5. Maps with Markers

Adding a marker:

```python
# Map of Canada
canada_map = folium.Maps(
	location=[56.130, -106.35], # center
	zoom_start=4				# zoom level
)

# Add a red marker to Ontario
# Create a feature group
ontario = folium.map.FeatureGroup()

# Style the feature group
ontario.add_child(
	folium.features.CircleMarker(
		[51.25, -85.32], radius=5,
		color="red", fill_color="Red"
	)
)

# Add the feature group to the map
canada_map.add_child(ontario)

# Label the marker
folium.Marker([51.25, -85.32], popup='Ontario').add_to(canada_map)
```

cluster markers superimposed onto a map in Folium using a marker cluster object.w

### 3.6. Choropleth Maps

**Definition**: A choropleth map is a thematic map in which areas are shaded or patterned in proportion to the measurement of the statistical variable being displayed on the map, such as population density or per capita income. The higher the measurement the darker the color



### LAB 1: Advanced Visualization Tools
### LAB 2: Creating Maps and Visualizing Geospatial Data
### QUIZ 1: Advanced Visualization Tools
### QUIZ 2: Quiz: Visualizing Geospatial Data

### Peer-review Assignment