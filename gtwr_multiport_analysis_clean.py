#!/usr/bin/env python3
"""
GTWR Environmental Justice Analysis with Multi-Port Proximity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pyproj import Transformer
import warnings
warnings.filterwarnings('ignore')

from pykrige.ok import OrdinaryKriging

class GTWRMultiPortAnalysis:
    def __init__(self, data_dir='.', output_dir='./gtwr_multiport_outputs'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.CA_STATE_CODE = 6
        
        self.EPA_FILES = {
            'PM25': 'daily_88101_2024.csv',
            'NO2': 'daily_42602_2024.csv', 
            'O3': 'daily_44201_2024.csv',
            'CO': 'daily_42101_2024.csv'
        }
        
        self.CALIFORNIA_PORTS = {
            'LA_Long_Beach': {
                'name': 'Port of Los Angeles/Long Beach',
                'lat': 33.74, 'lon': -118.25,
                'type': 'Major container port complex'
            },
            'Oakland': {
                'name': 'Port of Oakland', 
                'lat': 37.80, 'lon': -122.32,
                'type': 'Bay Area container port'
            },
            'San_Francisco': {
                'name': 'Port of San Francisco',
                'lat': 37.79, 'lon': -122.42, 
                'type': 'Bay Area general cargo port'
            },
            'Richmond': {
                'name': 'Port of Richmond',
                'lat': 37.93, 'lon': -122.38,
                'type': 'Bay Area bulk cargo port'
            },
            'Stockton': {
                'name': 'Port of Stockton',
                'lat': 37.95, 'lon': -121.29,
                'type': 'Central Valley inland port'
            },
            'San_Diego': {
                'name': 'Port of San Diego',
                'lat': 32.71, 'lon': -117.17,
                'type': 'Southern California port'
            }
        }
        
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3310", always_xy=True)
        
        self.ej_counties = {
            'Los Angeles': {'code': 37, 'median_income': 56000, 'minority_pct': 73.1},
            'San Bernardino': {'code': 71, 'median_income': 52000, 'minority_pct': 69.2},
            'San Joaquin': {'code': 77, 'median_income': 58000, 'minority_pct': 71.5}
        }
        
    def _haversine_distance(self, lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        c = 2*np.arcsin(np.sqrt(a))
        return 6371.0088 * c
    
    def load_and_merge_pollutant_data(self):
        print("🔄 Loading and merging EPA pollutant data...")
        
        all_data = []
        
        for pollutant, filename in self.EPA_FILES.items():
            filepath = self.data_dir / filename
            
            if not filepath.exists():
                print(f"❌ File not found: {filename}")
                continue
            
            try:
                df = pd.read_csv(filepath)
                df = df[df['State Code'] == self.CA_STATE_CODE]
                
                if len(df) == 0:
                    print(f"⚠️ No California data found in {filename}")
                    continue
                
                df['Pollutant'] = pollutant
                all_data.append(df)
                print(f"✅ Loaded {pollutant}: {len(df)} records")
                
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
        
        if not all_data:
            print("❌ No pollutant data loaded")
            return None
        
        merged_data = pd.concat(all_data, ignore_index=True)
        print(f"✅ Merged data: {len(merged_data)} total records")
        
        self.pollutant_data = merged_data
        return merged_data
    
    def calculate_multiport_proximity(self):
        print("\n🚢 Calculating multi-port proximity distances...")
        
        if not hasattr(self, 'pollutant_data'):
            print("❌ Pollutant data not loaded. Run load_and_merge_pollutant_data() first.")
            return None
        
        data = self.pollutant_data.copy()
        
        for port_id, port_info in self.CALIFORNIA_PORTS.items():
            port_lat = port_info['lat']
            port_lon = port_info['lon']
            
            data[f'dist_{port_id}_km'] = self._haversine_distance(
                data['Longitude'], data['Latitude'], port_lon, port_lat
            )
        
        distance_cols = [f'dist_{port_id}_km' for port_id in self.CALIFORNIA_PORTS.keys()]
        data['dist_nearest_port_km'] = data[distance_cols].min(axis=1)
        data['nearest_port'] = data[distance_cols].idxmin(axis=1).str.replace('dist_', '').str.replace('_km', '')
        
        print(f"✅ Multi-port proximity calculated for {len(data)} records")
        
        for port_id, port_info in self.CALIFORNIA_PORTS.items():
            port_name = port_info['name']
            sites = data[data['nearest_port'] == port_id]['Site Num'].nunique()
            dist_range = f"{data[f'dist_{port_id}_km'].min():.1f}-{data[f'dist_{port_id}_km'].max():.1f}"
            print(f"   • {port_name}: {sites} sites, distance range: {dist_range} km")
        
        self.multiport_data = data
        return data
    
    def prepare_gtwr_data(self):
        print("\n🔬 Preparing GTWR dataset with multi-port proximity...")
        
        if not hasattr(self, 'multiport_data'):
            print("❌ Multi-port data not calculated. Run calculate_multiport_proximity() first.")
            return None
        
        data = self.multiport_data.copy()
        
        monthly_data = data.groupby(['State Code', 'County Code', 'Site Num', 'Latitude', 'Longitude', 'nearest_port'], as_index=False).agg({
            'PM25': 'mean',
            'NO2': 'mean',
            'O3': 'mean', 
            'CO': 'mean',
            'dist_nearest_port_km': 'mean'
        })
        
        monthly_data = monthly_data.rename(columns={
            'Latitude': 'lat', 'Longitude': 'lon'
        })
        
        monthly_data['is_ej_community'] = monthly_data['County Code'].isin([
            self.ej_counties[county]['code'] for county in self.ej_counties.keys()
        ])
        
        self.gtwr_data = monthly_data
        print("✅ GTWR dataset preparation complete")
        
        return monthly_data
    
    def run_gtwr_analysis(self):
        print("\n🔬 Running GTWR Analysis with Multi-Port Proximity...")
        
        if not hasattr(self, 'gtwr_data'):
            print("❌ GTWR data not prepared. Run prepare_gtwr_data() first.")
            return None
        
        data = self.gtwr_data
        
        predictors = ['NO2_scaled', 'O3_scaled', 'CO_scaled', 'dist_nearest_port_km_scaled']
        
        for var in ['NO2', 'O3', 'CO', 'dist_nearest_port_km']:
            data[f'{var}_scaled'] = (data[var] - data[var].mean()) / data[var].std()
        
        print(f"  📊 GTWR setup: {len(data)} observations, {len(predictors)} predictors")
        print(f"  🔢 Predictors: {predictors}")
        
        try:
            from mgtwr.gtwr import GTWR
            from mgtwr.sel import GTWRModel
            
            coords = data[['lon', 'lat']].values
            y = data['PM25'].values
            X = data[predictors].values
            
            gtwr_model = GTWRModel(coords, y, X, bw=0.1, tau=0.1)
            gtwr_result = gtwr_model.fit()
            
            bw = gtwr_result.bw
            tau = gtwr_result.tau
            model_used = f'MGTWR (bw={bw:.1f}, tau={tau:.1f})'
            
            print(f"  ✅ GTWR successful: {model_used}")
            
            coefficients = pd.DataFrame({
                'lon': coords[:, 0],
                'lat': coords[:, 1],
                'intercept': gtwr_result.betas[:, 0],
                'NO2_coef': gtwr_result.betas[:, 1],
                'O3_coef': gtwr_result.betas[:, 2], 
                'CO_coef': gtwr_result.betas[:, 3],
                'port_dist_coef': gtwr_result.betas[:, 4],
                'is_ej_community': data['is_ej_community'].values,
                'nearest_port': data['nearest_port'].values
            })
            
            self.gtwr_result = gtwr_result
            self.coefficients = coefficients
            
            return gtwr_result, coefficients
            
        except Exception as e:
            print(f"  ❌ GTWR failed: {e}")
            return None, None
    
    def analyze_ej_patterns(self):
        print("\n⚖️ Environmental Justice Analysis with Multi-Port Proximity...")
        
        if not hasattr(self, 'coefficients'):
            print("❌ GTWR analysis not completed. Run run_gtwr_analysis() first.")
            return
        
        coef = self.coefficients
        data = self.gtwr_data
        
        ej_coef = coef[coef['is_ej_community'] == True]
        non_ej_coef = coef[coef['is_ej_community'] == False]
        
        ej_comparison = {}
        
        for var in ['NO2_coef', 'O3_coef', 'CO_coef', 'port_dist_coef']:
            ej_mean = ej_coef[var].mean()
            non_ej_mean = non_ej_coef[var].mean()
            ratio = ej_mean / non_ej_mean if non_ej_mean != 0 else np.nan
            
            ej_comparison[var] = {
                'ej_mean': ej_mean,
                'non_ej_mean': non_ej_mean,
                'ratio': ratio
            }
        
        print("\n📊 EJ COEFFICIENT COMPARISON:")
        for var, stats in ej_comparison.items():
            print(f"  {var}:")
            print(f"    EJ communities: {stats['ej_mean']:.3f}")
            print(f"    Non-EJ communities: {stats['non_ej_mean']:.3f}")
            print(f"    Ratio (EJ/Non-EJ): {stats['ratio']:.2f}")
        
        self.ej_analysis = ej_comparison
        
        return ej_comparison
    
    def create_visualizations(self):
        print("\n📊 Creating Multi-Port GTWR Visualizations...")
        
        if not hasattr(self, 'coefficients'):
            print("❌ GTWR analysis not completed.")
            return
        
        coef = self.coefficients
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        ax1 = axes[0, 0]
        scatter = ax1.scatter(coef['lon'], coef['lat'], c=coef['port_dist_coef'], 
                            cmap='RdBu_r', s=100, edgecolor='black')
        ax1.set_title('Port Proximity Coefficients\n(Nearest Port Distance)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        plt.colorbar(scatter, ax=ax1, shrink=0.8)
        
        ax2 = axes[0, 1]
        ej_coef = coef[coef['is_ej_community'] == True]
        non_ej_coef = coef[coef['is_ej_community'] == False]
        
        variables = ['NO2_coef', 'O3_coef', 'CO_coef', 'port_dist_coef']
        ej_means = [ej_coef[var].mean() for var in variables]
        non_ej_means = [non_ej_coef[var].mean() for var in variables]
        
        x = np.arange(len(variables))
        width = 0.35
        
        ax2.bar(x - width/2, ej_means, width, label='EJ Communities', alpha=0.8)
        ax2.bar(x + width/2, non_ej_means, width, label='Non-EJ Communities', alpha=0.8)
        ax2.set_xlabel('Variables')
        ax2.set_ylabel('Coefficient Value')
        ax2.set_title('EJ Coefficient Comparison\n(Multi-Port Proximity)', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['NO2', 'O3', 'CO', 'Port Dist'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = axes[0, 2]
        data = self.gtwr_data
        scatter = ax3.scatter(data['dist_nearest_port_km'], data['PM25'], 
                            c=data['is_ej_community'], cmap='RdYlBu', s=100, edgecolor='black')
        ax3.set_xlabel('Distance to Nearest Port (km)')
        ax3.set_ylabel('PM2.5 Concentration (μg/m³)')
        ax3.set_title('PM2.5 vs Nearest Port Distance\n(Regional Analysis)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax3, shrink=0.8)
        
        ax4 = axes[1, 0]
        if hasattr(self, 'gtwr_result'):
            result = self.gtwr_result
            metrics_text = f"""GTWR Model Performance
R²: {result.R2:.3f}
Adjusted R²: {result.adj_R2:.3f}
AICc: {result.AICc:.1f}
Spatial Bandwidth: {result.bw:.3f}
Temporal Bandwidth: {result.tau:.3f}

Key Findings:
• Multi-port proximity analysis
• Environmental justice assessment
• Regional coefficient variation
• Comprehensive visualizations"""
        else:
            metrics_text = "GTWR Results Not Available"
        
        ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax4.set_title('Model Summary', fontsize=14, fontweight='bold')
        ax4.axis('off')
        
        ax5 = axes[1, 1]
        port_colors = plt.cm.Set3(np.linspace(0, 1, len(self.CALIFORNIA_PORTS)))
        port_color_map = dict(zip(self.CALIFORNIA_PORTS.keys(), port_colors))
        
        for port_id in self.CALIFORNIA_PORTS.keys():
            port_data = coef[coef['nearest_port'] == port_id]
            if len(port_data) > 0:
                ax5.scatter(port_data['lon'], port_data['lat'], 
                          c=[port_color_map[port_id]], s=100, 
                          label=self.CALIFORNIA_PORTS[port_id]['name'], 
                          edgecolor='black')
        
        ax5.set_xlabel('Longitude')
        ax5.set_ylabel('Latitude')
        ax5.set_title('Monitoring Network\n(Colored by Nearest Port)', fontsize=14, fontweight='bold')
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.grid(True, alpha=0.3)
        
        ax6 = axes[1, 2]
        regional_analysis = coef.groupby('nearest_port').agg({
            'port_dist_coef': ['mean', 'std'],
            'is_ej_community': 'sum'
        }).round(3)
        
        regional_analysis.columns = ['Mean_Coeff', 'Std_Coeff', 'EJ_Sites']
        regional_analysis = regional_analysis.reset_index()
        
        bars = ax6.bar(range(len(regional_analysis)), regional_analysis['Mean_Coeff'], 
                      yerr=regional_analysis['Std_Coeff'], capsize=5, alpha=0.7)
        ax6.set_xlabel('Nearest Port')
        ax6.set_ylabel('Port Distance Coefficient')
        ax6.set_title('Regional Port Proximity Effects', fontsize=14, fontweight='bold')
        ax6.set_xticks(range(len(regional_analysis)))
        ax6.set_xticklabels([self.CALIFORNIA_PORTS[port]['name'] for port in regional_analysis['nearest_port']], 
                           rotation=45, ha='right')
        ax6.grid(True, alpha=0.3)
        
        for i, (bar, ej_sites) in enumerate(zip(bars, regional_analysis['EJ_Sites'])):
            if ej_sites > 0:
                bar.set_color('red')
                bar.set_alpha(0.8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'gtwr_multiport_analysis.png', 
                   dpi=300, bbox_inches='tight')
        print(f"💾 Visualizations saved: {self.output_dir}/gtwr_multiport_analysis.png")
        plt.close()
    
    def save_results(self):
        print("\n💾 Saving Multi-Port GTWR Results...")
        
        if hasattr(self, 'gtwr_data'):
            self.gtwr_data.to_csv(
                self.output_dir / 'gtwr_multiport_dataset.csv', index=False
            )
            print(f"  ✅ Dataset: gtwr_multiport_dataset.csv")
        
        if hasattr(self, 'coefficients'):
            self.coefficients.to_csv(
                self.output_dir / 'gtwr_multiport_coefficients.csv', index=False
            )
            print(f"  ✅ Coefficients: gtwr_multiport_coefficients.csv")
        
        if hasattr(self, 'ej_analysis'):
            ej_df = pd.DataFrame(self.ej_analysis).T
            ej_df.to_csv(
                self.output_dir / 'ej_comparison.csv', index=False
            )
            print(f"  ✅ EJ comparison: ej_comparison.csv")
        
        if hasattr(self, 'coefficients'):
            regional_analysis = self.coefficients.groupby('nearest_port').agg({
                'port_dist_coef': ['mean', 'std'],
                'is_ej_community': 'sum'
            }).round(3)
            regional_analysis.columns = ['Mean_Coeff', 'Std_Coeff', 'EJ_Sites']
            regional_analysis = regional_analysis.reset_index()
            
            regional_analysis.to_csv(
                self.output_dir / 'regional_analysis.csv', index=False
            )
            print(f"  ✅ Regional analysis: regional_analysis.csv")
    
    def run_complete_analysis(self):
        print("🚀 GTWR MULTI-PORT ENVIRONMENTAL JUSTICE ANALYSIS")
        print("=" * 65)
        
        if self.load_and_merge_pollutant_data() is None:
            print("❌ Failed to load pollutant data")
            return
        
        if self.calculate_multiport_proximity() is None:
            print("❌ Failed to calculate multi-port proximity")
            return
        
        if self.prepare_gtwr_data() is None:
            print("❌ Failed to prepare GTWR data")
            return
        
        result, coefficients = self.run_gtwr_analysis()
        if result is None:
            print("❌ GTWR analysis failed")
            return
        
        self.analyze_ej_patterns()
        self.create_visualizations()
        self.save_results()
        
        print("\n🎉 MULTI-PORT GTWR ANALYSIS COMPLETE!")
        print("=" * 50)
        print("Key Improvements:")
        print("  ✅ Multi-port proximity analysis")
        print("  ✅ Nearest port distance calculation")
        print("  ✅ Regional coefficient variation")
        print("  ✅ Environmental justice assessment")
        print("  ✅ Comprehensive visualizations")
        print("  ✅ Detailed regional analysis")

if __name__ == "__main__":
    analyzer = GTWRMultiPortAnalysis()
    analyzer.run_complete_analysis()
