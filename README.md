# Air Pollution & Environmental Justice Research  
**Investigating spatial disparities in pollution exposure near California’s major ports**

## Overview

This project explores whether low-income and minority communities in California experience disproportionately high exposure to harmful air pollutants, particularly fine particulate matter (PM2.5), due to proximity to major ports. We combine geospatial analysis, statistical modeling, and open environmental data to better understand the intersection of environmental justice and public health.

## Research Questions

- Do communities near major California ports experience higher levels of PM2.5?
- Are these communities disproportionately composed of low-income or minority populations?
- How do these patterns vary over space and time?

## Data & Sources

- **EPA Air Quality System (AQS)**: Hourly and daily PM2.5 readings  
- **California Environmental Protection Agency (CalEPA)**: Disadvantaged community indices  
- **U.S. Census Bureau**: Demographics, income, and housing statistics  
- **Port geolocations**: Long Beach, Los Angeles, Oakland, etc.  
- Data was cleaned, standardized, and joined using Python (`pandas`, `geopandas`, `shapely`, `numpy`)

## Methodology Summary

### 1. **Data Cleaning & Transformation**  
- Filtered EPA data for completeness  
- Merged pollution data with census and CalEnviroScreen scores  
- Applied spatial joins to map each community’s proximity to ports

### 2. **Geographically & Temporally Weighted Regression (GTWR)**  
- Used GTWR to model how PM2.5 levels relate to port proximity, adjusting for time and location  
- Allowed for localized relationships that vary across different regions and years

### 3. **Kriging Interpolation**  
- Interpolated pollution concentrations between monitoring stations  
- Produced high-resolution pollution heatmaps with uncertainty estimates  
- Powered by `pykrige` and `scikit-learn`

### 4. **Environmental Justice Analysis**  
- Overlayed GTWR and kriging outputs with demographic data  
- Identified areas where pollution burden intersects with social vulnerability

## Key Findings

- **Higher PM2.5 levels** were consistently detected in communities within 10 km of major ports  
- These communities were **significantly more likely to be low-income and majority non-white**  
- Spatial models revealed **regional disparities** even after adjusting for population density and industrial zoning  
- Findings support the need for targeted environmental policies and emission-reduction measures in port-adjacent communities
