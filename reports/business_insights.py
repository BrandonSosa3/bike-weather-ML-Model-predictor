import pandas as pd
import matplotlib.pyplot as plt

def generate_business_recommendations():
    """Generate actionable business insights from our model"""
    
    # Here we have a list of dictionaries, each dict has 3 parts:
    # 1. Finding = the high level observation
    # 2. Insight = the detailed explanation, often with numbers or stats
    # 3. Action = what the business should do about it
    # So basically instead of raw model numbers, this tells a story for a human
    findings = [
        {
            'finding': 'Time Patterns Dominate Weather',
            'insight': 'Hour of day explains 48.6% of usage variance',
            'action': 'Focus operational planning on time patterns first, weather second'
        },
        {
            'finding': 'Weather Affects User Types Differently', 
            'insight': 'Commuters are 5x more weather-resistant than recreational riders',
            'action': 'Different marketing and pricing strategies for weekdays vs weekends'
        },
        {
            'finding': 'Rush Hour Premium Opportunity',
            'insight': 'Perfect weather rush hour can hit 550+ bikes/hour',
            'action': 'Dynamic pricing and station rebalancing during peak hours'
        },
        {
            'finding': 'Weekend Weather Sensitivity',
            'insight': 'Heavy rain drops weekend usage by 86%',
            'action': 'Weather-based staffing and maintenance scheduling'
        }
    ]
    
    # enumerate(findings, 1) → gives both the index (i) starting at 1, and the actual dictionary (item).
    for i, item in enumerate(findings, 1):
        print(f"\n{i}. 🔍 {item['finding']}")
        print(f"   Insight: {item['insight']}")
        print(f"   Action: {item['action']}")
    
    print(f"\nOPERATIONAL RECOMMENDATIONS:")
    
    recommendations = [
        "Station Rebalancing: Focus on 7-9am and 5-7pm windows",
        "Weather Response: Reduce weekend staffing when rain forecast",
        "Dynamic Pricing: Premium rates during perfect weather rush hours",
        "Maintenance Windows: Schedule during early morning (6am) low usage",
        "User Notifications: Weather-based usage alerts for weekend riders",
        "Capacity Planning: Size stations based on rush hour + weather premium"
    ]
    
    for rec in recommendations:
        print(f"  • {rec}")

def create_final_summary_visualization():
    """Create a final summary chart of our discoveries"""
    
    print(f"\nCreating final summary visualization...")
    
    # Import the prediction function from our interface
    from models.prediction_interface import create_prediction_interface
    
    # Get the actual prediction function
    # calls create_prediction_interface() which returns 
    # 1. predict_func -> a function you can call with custom inputs (hour, temp, weather, etc) to get a prediction
    # 2. scenario_results -> a prebuilt list of scenarios with predictions
    predict_func, scenario_results = create_prediction_interface()
    
    # Extract scenarios and their predictions dynamically
    scenarios = []
    predictions = []
    colors = ['green', 'orange', 'lightgreen', 'red', 'lightblue']
    
    for result in scenario_results:
        # Clean up scenario names for display
        scenario_name = result['scenario'].replace('🌞 ', '').replace('🌧️ ', '').replace('☀️ ', '').replace('🌨️ ', '').replace('🌅 ', '')
        # Split long names into multiple lines
        scenario_name = scenario_name.replace(' (', '\n(').replace(' ', '\n', 1)
        
        scenarios.append(scenario_name)
        predictions.append(result['prediction'])
    
    plt.figure(figsize=(12, 8))
    
    # Create bar chart with actual predictions
    bars = plt.bar(scenarios, predictions, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Bike Share Usage Predictions by Scenario', 
              fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Predicted Bikes per Hour', fontsize=12)
    plt.xlabel('Weather & Time Scenarios', fontsize=12)
    
    # Add horizontal average line using actual average
    avg_prediction = sum(predictions) / len(predictions)
    plt.axhline(y=avg_prediction, color='black', linestyle='--', alpha=0.5, 
                label=f'Average: {avg_prediction:.0f} bikes/hour')
    
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the chart
    plt.savefig('final_project_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Final summary chart saved as: final_project_summary.png")

if __name__ == "__main__":
    generate_business_recommendations()
    create_final_summary_visualization()