// Initialize total asset value
let initialAssetValue = 19991121;

const exposureMatrixData = [
    ["Asset_300", -0.100, 1.146, 2.248, -0.364, -0.045, -0.295],
    ["Asset_1000", -0.006, 1.141, 3.054, 0.367, 0.054, -0.217],
    ["Asset_cyb", -0.711, 1.831, 3.242, -0.295, 0.026, -0.163],
    ["Asset_500", -0.145, 1.278, 2.623, 0.097, -0.027, -0.256],
    ["Asset_guozhai", -0.106, -0.011, 0.008, -0.098, -0.016, -0.011],
    ["Asset_qiyezhai", -0.044, 0.003, 0.040, -0.030, -0.023, -0.016],
    ["Asset_gongyepin", 0.637, 2.138, 2.369, -0.721, 0.380, -0.529],
    ["Asset_nongchanpin", 0.307, 1.155, 1.593, -0.616, 0.610, -0.641],
    ["Asset_tjin", -0.442, 0.200, -0.942, -0.003, -0.222, -0.038],
];


// Utility function to load CSV files
async function loadDataFrame(filePath) {
    return await dfd.readCSV(filePath);
}

// Calculate percentage changes
function calculatePercentageChange(current, initial) {
    return ((current - initial) / initial) * 100;
}

// Format number with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Function to populate the exposure matrix
function populateExposureMatrix(tableId, data) {
    const table = document.getElementById(tableId);
    const tbody = table.querySelector("tbody");

    // Clear existing rows (if any)
    tbody.innerHTML = "";

    // Populate rows dynamically
    data.forEach((row) => {
        const tr = document.createElement("tr");
        row.forEach((cell) => {
            const td = document.createElement("td");
            td.textContent = typeof cell === "number" ? cell.toFixed(3) : cell; // Format to 3 decimal places
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
}

function joinAndFill(assetPricesDf, allocationDf) {
    // Step 1: Parse dates into proper datetime format
    const assetPricesDfParsed = assetPricesDf.sortValues("Date", { ascending: true });
    const allocationDfParsed = allocationDf.sortValues("Date", { ascending: true });

    // Step 2: Perform inner join on the "Date" column
    joinedDf = dfd.merge({
        left: allocationDfParsed,
        right: assetPricesDfParsed,
        on: ["Date"],
        how: "left",
    });

    joinedDf = joinedDf.dropNa(1);


    const filledDf = joinedDf.sortValues("Date", { ascending: true });

    return filledDf;
}

function calculateYield() {
    y = now - old / old
}

function calculateYieldOverTime(assetPricesDf, weightsDf, initialAssetValue) {
    const combinedDf = joinAndFill(assetPricesDf, weightsDf);

    const dates = assetPricesDf.column("Date").values;
    const weights = combinedDf.iloc({ columns: ["1:6"] }).values;
    const prices = combinedDf.iloc({ columns: ["7:12"] }).values;

    const yields = [1]; // Array to store daily yields (percentage change)
    const normalizedYields = [Array(prices[0].length).fill(1)]; // Array for normalized yields
    let previousAssetValue = prices[0];

    for (let i = 1; i < prices.length; i++) {

        let dailyYields = Array(prices[i].length).fill(0);
        let yield = 0;
        for (let j = 0; j < prices[i].length; j++) {
            if (previousAssetValue[j] === 0) {
                dailyYields[j] = 0;
            }
            else {
                dailyYields[j] = (prices[i][j]- previousAssetValue[j]) / previousAssetValue[j];
            }
        }
        
        let normalizedYield = Array(prices[i].length).fill(1);
        for (let j = 0; j < prices[i].length; j++) {
            normalizedYield[j] = normalizedYields[i-1][j] * (1 + dailyYields[j]);
        }
        normalizedYields.push(normalizedYield);

        for (let j = 0; j < prices[i].length; j++) {
            yield += normalizedYield[j] * weights[i][j];
        }
        // Push to the yields array
        yields.push(yield);

        // Update the previous asset value for the next day
        previousAssetValue = prices[i];
    }

    return { dates, yields };
}

function getIndividualYields(assetPricesDf) {
    const df = assetPricesDf.dropNa(1);
    const dates = df.column("Date").values
    let yields = {};
    for (let c = 1; c < df.columns.length; c++) {
        if (c !== "Date") {
            const value = df.iloc({ columns: [c] }).values;
            let yield = [1];
            for (let i = 1; i < value.length; i++) {
                if (value[i - 1] === 0)
                {
                    yield.push(yield[i-1]);
                }
                else
                {
                    yield.push(yield[i-1] * (1 + (value[i] - value[i - 1])/value[i - 1]));
                }
            }

            yields[df.columns[c]] = yield;
        }
    }
    return { dates, yields };
}


function displayYieldOverTime(yieldData, individualYields) {
    const transformModelData = (data) =>
        data.dates.map((date, i) => ({ date, yield: data.yields[i] }));

    const modelData = transformModelData(yieldData[0]);
    const baselineData = transformModelData(yieldData[1]);

    const individualLines = Object.keys(individualYields.yields).map((key) => ({
        name: key,
        values: individualYields.dates.map((date, i) => ({
            date,
            yield: individualYields.yields[key][i],
        })),
    }));

    const data = [
        { name: "Model Yield", values: modelData },
        { name: "Baseline Yield", values: baselineData },
        ...individualLines,
    ];


    // Parse dates
    const parseDate = d3.timeParse("%Y-%m-%d");

    // Chart dimensions
    const width = 800;
    const height = 400;
    const margin = { top: 20, right: 30, bottom: 50, left: 50 };

    // SVG container
    const svg = d3
        .select("#yield-over-time")
        .attr("width", width)
        .attr("height", height);

    // Chart area
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    const g = svg
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3
        .scaleTime()
        .domain([
            d3.min(data, (d) => d3.min(d.values, (v) => parseDate(v.date))),
            d3.max(data, (d) => d3.max(d.values, (v) => parseDate(v.date))),
        ])
        .range([0, chartWidth]);

    const yScale = d3
        .scaleLinear()
        .domain([
            d3.min(data, (d) => d3.min(d.values, (v) => v.yield)),
            d3.max(data, (d) => d3.max(d.values, (v) => v.yield)),
        ])
        .range([chartHeight, 0]);

    // Axes
    const xAxis = d3.axisBottom(xScale).ticks(5);
    const yAxis = d3
        .axisLeft(yScale)
        .ticks(10, ",.1s") // Format ticks for log scale 
        .tickFormat((d) => d3.format(",.1f")(d));

    g.append("g")
        .attr("class", "x-axis")
        .attr("transform", `translate(0,${chartHeight})`)
        .call(xAxis);

    g.append("g").attr("class", "y-axis").call(yAxis);

    // Line generator
    const line = d3
        .line()
        .x((d) => xScale(parseDate(d.date)))
        .y((d) => yScale(d.yield));

    // Color scale
    const color = d3.scaleOrdinal(d3.schemeCategory10);

    // Draw lines
    g.selectAll(".line")
        .data(data)
        .join("path")
        .attr("class", "line")
        .attr("d", (d) => line(d.values))
        .attr("stroke", (d) => color(d.name))
        .attr("fill", "none")
        .attr("stroke-width", 2);

    // Add legend
    const legend = svg
        .append("g")
        .attr("transform", `translate(${margin.left},${height + margin.top})`); // Adjust to fit below the chart

    data.forEach((d, i) => {
        const xOffset = (i % 3) * 150; // Items per row
        const yOffset = Math.floor(i / 3) * 20; // Row height

        legend
            .append("circle")
            .attr("cx", xOffset)
            .attr("cy", yOffset)
            .attr("r", 5)
            .attr("fill", color(d.name));

        legend
            .append("text")
            .attr("x", xOffset + 10)
            .attr("y", yOffset + 5)
            .attr("class", "legend")
            .text(d.name);
    });

    // Resize SVG dynamically for legend height
    const legendHeight = Math.ceil(data.length / 3) * 20 + margin.bottom;
    svg.attr("height", height + legendHeight);
}

function getAllocationWeights(allocationDf) {
    const df = allocationDf.dropNa(1);
    const dates = df.column("Date").values
    let weights = {};
    for (let c = 1; c < df.columns.length; c++) {
        if (c !== "Date") {
            const weight = df.iloc({ columns: [c] }).values;
            weights[df.columns[c]] = weight;
        }
    }
    return { dates, weights };
}

function displayAllocationWeights(weights) {
    const individualLines = Object.keys(weights.weights).map((key) => ({
        name: key,
        values: weights.dates.map((date, i) => ({
            date,
            yield: weights.weights[key][i],
        })),
    }));

    const data = [
        ...individualLines,
    ];

    // Parse dates
    const parseDate = d3.timeParse("%Y-%m-%d");

    // Chart dimensions
    const width = 800;
    const height = 400;
    const margin = { top: 20, right: 30, bottom: 50, left: 50 };

    // SVG container
    const svg = d3
        .select("#allocation-over-time")
        .attr("width", width)
        .attr("height", height);

    // Chart area
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    const g = svg
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3
        .scaleTime()
        .domain([
            d3.min(data, (d) => d3.min(d.values, (v) => parseDate(v.date))),
            d3.max(data, (d) => d3.max(d.values, (v) => parseDate(v.date))),
        ])
        .range([0, chartWidth]);

    const yScale = d3
        .scaleLinear()
        .domain([
            0, 1
        ])
        .range([chartHeight, 0]);

    // Axes
    const xAxis = d3.axisBottom(xScale).ticks(5);
    const yAxis = d3
        .axisLeft(yScale)
        .ticks(10, ",.1s") // Format ticks for log scale 
        .tickFormat((d) => d3.format(",.1f")(d));

    g.append("g")
        .attr("class", "x-axis")
        .attr("transform", `translate(0,${chartHeight})`)
        .call(xAxis);

    g.append("g").attr("class", "y-axis").call(yAxis);

    // Line generator
    const line = d3
        .line()
        .x((d) => xScale(parseDate(d.date)))
        .y((d) => yScale(d.yield));

    // Color scale
    const color = d3.scaleOrdinal(d3.schemeCategory10);

    // Draw lines
    g.selectAll(".line")
        .data(data)
        .join("path")
        .attr("class", "line")
        .attr("d", (d) => line(d.values))
        .attr("stroke", (d) => color(d.name))
        .attr("fill", "none")
        .attr("stroke-width", 2);

    // Add legend
    const legend = svg
        .append("g")
        .attr("transform", `translate(${margin.left},${height + margin.top})`); // Adjust to fit below the chart

    data.forEach((d, i) => {
        const xOffset = (i % 3) * 150; // Items per row
        const yOffset = Math.floor(i / 3) * 20; // Row height

        legend
            .append("circle")
            .attr("cx", xOffset)
            .attr("cy", yOffset)
            .attr("r", 5)
            .attr("fill", color(d.name));

        legend
            .append("text")
            .attr("x", xOffset + 10)
            .attr("y", yOffset + 5)
            .attr("class", "legend")
            .text(d.name);
    });

    // Resize SVG dynamically for legend height
    const legendHeight = Math.ceil(data.length / 3) * 20 + margin.bottom;
    svg.attr("height", height + legendHeight);
}
    

function calculateCorrelationMatrix(df) {
    const numericDf = df.drop({ columns: ["Date"] });

    const columns = numericDf.columns;
    const data = numericDf.values;

    // Initialize correlation matrix
    const correlationMatrix = Array(columns.length)
        .fill(0)
        .map(() => Array(columns.length).fill(0));

    // Compute correlations
    for (let i = 0; i < columns.length; i++) {
        for (let j = 0; j < columns.length; j++) {
            const x = data.map((row) => row[i]); // Column i
            const y = data.map((row) => row[j]); // Column j
            correlationMatrix[i][j] = ss.sampleCorrelation(x, y); // Compute correlation
        }
    }

    // Convert to DataFrame
    return new dfd.DataFrame(correlationMatrix, {
        columns: columns,
        index: columns,
    });
}

function populateCorrelationMatrix(tableId, CovDf) {
    const columns = ["", ...CovDf.columns]; // "Asset" represents the first column label
    const values = CovDf.values;
    const index = CovDf.index; // Extract row labels (index)

    // Create the HTML table
    let tableHtml = "<table>";

    // Add table headers
    tableHtml += "<thead><tr>";
    columns.forEach((col) => {
        tableHtml += `<th>${col}</th>`;
    });
    tableHtml += "</tr></thead>";

    // Add table rows
    tableHtml += "<tbody>";
    values.forEach((row, rowIndex) => {
        tableHtml += "<tr>";
        tableHtml += `<td>${index[rowIndex]}</td>`; // Add the row label (index) as the first column
        row.forEach((cell) => {
            tableHtml += `<td>${cell.toFixed(3)}</td>`; // Format values to 3 decimal places
        });
        tableHtml += "</tr>";
    });
    tableHtml += "</tbody>";

    tableHtml += "</table>";

    // Insert the HTML table into the specified container
    document.getElementById(tableId).innerHTML = tableHtml;
}

// Main function to load data and perform calculations
async function main() {
    // Load data from the "data" directory
    const [assetPricesDf, riskParityDf, riskNeutralAllocations, factorsDf] = await Promise.all([
        loadDataFrame("./data/assets.csv"),
        loadDataFrame("./data/risk_parity_allocations.csv"),
        loadDataFrame("./data/risk_neutral_allocations.csv"),
        loadDataFrame("./data/factors.csv"),
    ]);

    // Calculate current yield based on risk parity
    const modelYieldData = calculateYieldOverTime(assetPricesDf, riskParityDf, initialAssetValue);
    const baselineYieldData = calculateYieldOverTime(assetPricesDf, riskNeutralAllocations, initialAssetValue);
    const individualYields = getIndividualYields(assetPricesDf);

    // Calculate day-before gain
    displayYieldOverTime([modelYieldData, baselineYieldData], individualYields);

    // Display allocation weights
    const weights = getAllocationWeights(riskParityDf);
    displayAllocationWeights(weights);

    populateExposureMatrix("exposure-matrix", exposureMatrixData);

    const assetCov = calculateCorrelationMatrix(assetPricesDf);
    populateCorrelationMatrix("asset-covariance-matrix", assetCov);

    const factorCov = calculateCorrelationMatrix(factorsDf);
    populateCorrelationMatrix("factor-covariance-matrix", factorCov);
}


document.addEventListener("DOMContentLoaded", () => {
    main();
});