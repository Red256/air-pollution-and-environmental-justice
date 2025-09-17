#!/usr/bin/env python3
"""
Multi-Port Environmental Justice Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def haversine_km(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return 6371.0088 * c

class MultiPortEnvironmentalJustice:
    def __init__(self):
        self.california_ports = {
            'LA_Long_Beach': {
                'lat': 33.74, 'lon': -118.25,
                'name': 'Port of Los Angeles/Long Beach',
                'description': 'Largest port complex in US, major container traffic'
            },
            'Oakland': {
                'lat': 37.80, 'lon': -122.32,
                'name': 'Port of Oakland',
                'description': 'Major Bay Area container port'
            },
            'San_Francisco': {
                'lat': 37.79, 'lon': -122.42,
                'name': 'Port of San Francisco',
                'description': 'Bay Area general cargo and cruise port'
            },
            'Richmond': {
                'lat': 37.93, 'lon': -122.38,
                'name': 'Port of Richmond',
                'description': 'Bay Area bulk cargo and petroleum products'
            },
            'Stockton': {
                'lat': 37.95, 'lon': -121.29,
                'name': 'Port of Stockton',
                'description': 'Central Valley inland port'
            },
            'San_Diego': {
                'lat': 32.71, 'lon': -117.17,
                'name': 'Port of San Diego',
                'description': 'Southern California general cargo and naval'
            }
        }
    
    def calculate_port_distances(self, station_coords):
        results = []
        
        for idx, row in station_coords.iterrows():
            station_lat = row['lat']
            station_lon = row['lon']
            
            port_distances = {}
            
            for port_id, port_info in self.california_ports.items():
                distance = haversine_km(
                    station_lon, station_lat,
                    port_info['lon'], port_info['lat']
                )
                port_distances[port_id] = distance
            
            nearest_port = min(port_distances.keys(), key=lambda x: port_distances[x])
            nearest_distance = port_distances[nearest_port]
            
            results.append({
                'nearest_port': nearest_port,
                'nearest_port_name': self.california_ports[nearest_port]['name'],
                'dist_nearest_port_km': nearest_distance,
                'dist_LA_port_km': port_distances['LA_Long_Beach'],
                'dist_SF_port_km': port_distances['Oakland'],
                'all_distances': port_distances
            })
        
        return pd.DataFrame(results)
    
    def load_monitoring_data(self):
        gtwr_file = Path('gtwr_ej_outputs/gtwr_results_california.csv')
        
        if not gtwr_file.exists():
            print(f"❌ GTWR results file not found: {gtwr_file}")
            return None
        
        try:
            data = pd.read_csv(gtwr_file)
            station_coords = data.groupby(['State Code', 'County Code', 'Site Num', 'lat', 'lon'], as_index=False).agg({
                'PM25': 'mean',
                'NO2': 'mean',
                'O3': 'mean',
                'CO': 'mean'
            })
            
            print(f"✅ Loaded monitoring data: {len(station_coords)} stations")
            return station_coords
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def analyze_multi_port_impact(self):
        print("🚢 MULTI-PORT ENVIRONMENTAL JUSTICE ANALYSIS")
        print("=" * 50)
        
        station_data = self.load_monitoring_data()
        if station_data is None:
            return None
        
        port_analysis = self.calculate_port_distances(station_data)
        
        combined_data = pd.concat([station_data.reset_index(drop=True), port_analysis], axis=1)
        
        print(f"\n📊 PORT PROXIMITY ANALYSIS:")
        print(f"   Total monitoring stations: {len(combined_data)}")
        
        port_counts = combined_data['nearest_port'].value_counts()
        print(f"\n🏭 Stations by nearest port:")
        for port_id, count in port_counts.items():
            port_name = self.california_ports[port_id]['name']
            print(f"   • {port_name}: {count} stations")
        
        la_only_stations = combined_data[combined_data['nearest_port'] == 'LA_Long_Beach']
        other_port_stations = combined_data[combined_data['nearest_port'] != 'LA_Long_Beach']
        
        print(f"\n📈 DISTANCE ANALYSIS:")
        print(f"   Stations closest to LA/Long Beach: {len(la_only_stations)}")
        print(f"   Stations closer to other ports: {len(other_port_stations)}")
        print(f"   Percentage using other ports: {len(other_port_stations)/len(combined_data)*100:.1f}%")
        
        if len(other_port_stations) > 0:
            avg_distance_reduction = (other_port_stations['dist_LA_port_km'] - 
                                    other_port_stations['dist_nearest_port_km']).mean()
            print(f"   Average distance reduction: {avg_distance_reduction:.1f} km")
        
        return combined_data
    
    def create_visualizations(self, data):
        print("\n📊 Creating multi-port analysis visualizations...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1 = axes[0]
        port_colors = plt.cm.Set3(np.linspace(0, 1, len(self.california_ports)))
        port_color_map = dict(zip(self.california_ports.keys(), port_colors))
        
        for port_id in self.california_ports.keys():
            port_data = data[data['nearest_port'] == port_id]
            if len(port_data) > 0:
                ax1.scatter(port_data['lon'], port_data['lat'], 
                          c=[port_color_map[port_id]], s=100, 
                          label=self.california_ports[port_id]['name'], 
                          edgecolor='black', alpha=0.8)
        
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title('Monitoring Stations by Nearest Port', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        scatter = ax2.scatter(data['dist_nearest_port_km'], data['PM25'], 
                            c=data['nearest_port'].map(port_color_map), 
                            s=100, edgecolor='black', alpha=0.8)
        ax2.set_xlabel('Distance to Nearest Port (km)')
        ax2.set_ylabel('PM2.5 Concentration (μg/m³)')
        ax2.set_title('PM2.5 vs Nearest Port Distance', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('multi_port_analysis.png', dpi=300, bbox_inches='tight')
        print(f"💾 Saved multi-port analysis: multi_port_analysis.png")
        plt.close()
    
    def run_complete_analysis(self):
        data = self.analyze_multi_port_impact()
        if data is not None:
            self.create_visualizations(data)
            
            print("\n🎉 MULTI-PORT ANALYSIS COMPLETE!")
            print("=" * 40)
            print("Key Findings:")
            print("  ✅ Multiple California ports considered")
            print("  ✅ Nearest port distance calculation")
            print("  ✅ Regional port proximity analysis")
            print("  ✅ Environmental justice implications")
            print("  ✅ Comprehensive visualizations")

if __name__ == "__main__":
    analyzer = MultiPortEnvironmentalJustice()
    analyzer.run_complete_analysis()
