# Earthquake relocation using 1D velocity model
Earthquake events are a great tool for studying the deep structures and the forces acting on them. But to use earthquakes for such applications, we need to know the locations, both on surface and at depth, of these events as accurately as possible. Otherwise, all subsequent interpretations will be greatly flawed. This location determination is dependent on the velocity structure of the earth. 

In this project, I am focusing on the Mendocino Triple-Junction (MTJ) in Northern California, where three tectonic plates meet: the Gorda, Pacific, and North American plates. Consequently, this region experiences a high frequency of seismic activity. Figure 1 illustrates a record of 17,961 earthquakes with a magnitude of 2.0 or greater, spanning from January 1, 1970, to August 31, 2023.

The primary objective of this project is to isolate earthquakes occurring above the subducting plate interface, using location data from the USGS earthquake catalog (https://earthquake.usgs.gov/earthquakes/search/) and relocate them as accurately possible. My interest is in investigating how the stress field varies both spatially (horizontally and vertically) and temporally. Figure 2 presents an east-west cross-section displaying earthquake depths. Noticeably, several earthquakes have been assigned fixed depths of either 5, 10, or 15 km due to limited constraints, underscoring the importance of earthquake relocation in this context. As my focus on a more specific geographic area compared to the USGS, I can employ local 3D velocity models for more accurate relocation. For the purposes of this project, I will initially use a generalized 1D velocity model to develop a thorough understanding of the relocation process and its correct application. Subsequently, I plan to implement a 3D model in my ongoing research.

![image](https://github.com/arif-geo/earth_sci_data_analysis_arif/assets/82399285/fbdab256-aa35-4852-a65a-b53972ef97a1)

Figure 1: Earthquake locations in the MTJ (blue dots) and slab depth (red-blue area) from Hayes (2018).

![image](https://github.com/arif-geo/earth_sci_data_analysis_arif/assets/82399285/57015de9-ac56-4d2f-91d2-b57f18cce841)

Figure 2: West-East depth cross section showing earthquake depths for all earthquakes in the study area. 

Currently, I am in the process of segregating earthquakes based on subduction slab depth and acquiring seismic data continuously for 24 hours a day from stations within my study area, spanning from January 1, 2008, to December 31, 2022. My goal is to complete the data processing steps within the next two weeks and commence the relocation procedure from the second week of November.


Reference: \
Hayes, G., 2018, Slab2 - A Comprehensive Subduction Zone Geometry Model: U.S. Geological Survey data release, https://doi.org/10.5066/F7PV6JNV.

