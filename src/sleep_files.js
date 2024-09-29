import React, { useEffect, useState } from 'react';
import { Line } from 'react-chartjs-2';
import * as d3 from 'd3';

const SleepPlot = () => {
  const [sleepData, setSleepData] = useState([]);
  const [plotData, setPlotData] = useState({});
  const [varianceData, setVarianceData] = useState([]);

  // Load and parse the JSON data (assuming local data for now)
  useEffect(() => {
    d3.json('/path/to/sleep_data.json').then((data) => {
      const parsedData = parseSleepData(data);
      const transformedData = transformToPlotData(parsedData);
      setSleepData(parsedData);
      setPlotData(transformedData);
      setVarianceData(calculateVariance(transformedData));
    });
  }, []);

  // Function to parse sleep data
  const parseSleepData = (data) => {
    return data.map((entry) => ({
      start: new Date(entry.startTime),
      stop: new Date(entry.endTime),
      duration: parseFloat(entry.duration) / 3600,
    }));
  };

  // Transform data into format for the Line chart
  const transformToPlotData = (data) => {
    const labels = data.map((d) => d.start.toLocaleTimeString());
    const sleepDurations = data.map((d) => d.duration);
    return {
      labels: labels,
      datasets: [
        {
          label: 'Sleep Duration (hours)',
          data: sleepDurations,
          fill: false,
          backgroundColor: 'rgba(75,192,192,1)',
          borderColor: 'rgba(75,192,192,0.4)',
        },
      ],
    };
  };

  // Function to calculate variance
  const calculateVariance = (data) => {
    const durations = data.datasets[0].data;
    const mean = d3.mean(durations);
    return durations.map((value) => Math.pow(value - mean, 2));
  };

  return (
    <div>
      <h2>Sleep Duration Plot</h2>
      <Line data={plotData} />
      <h2>Variance Data</h2>
      <Line
        data={{
          labels: sleepData.map((_, idx) => `Day ${idx + 1}`),
          datasets: [
            {
              label: 'Variance',
              data: varianceData,
              backgroundColor: 'rgba(255,99,132,0.2)',
              borderColor: 'rgba(255,99,132,1)',
              borderWidth: 1,
              fill: false,
            },
          ],
        }}
      />
    </div>
  );
};

export default SleepPlot;
