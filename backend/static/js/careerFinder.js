const cardGrid = document.getElementById('card-grid');

var color = d3.scale.ordinal().range(["#CC333F","#00A0B0"]);

var radarChartOptions = {
    w: '300',
    h: '300',
    maxValue: 1,
    levels: 5,
    roundStrokes: false,
    color: color
};

function drawSpiderChart(cardEltId, data) {
    let svgId = "." + cardEltId;

    let clean_data = [];
    for (let i = 0; i < 8; i++) {
        axis_name = data[i][0];
        axis_name = axis_name.replace(/_/g, ' ');
        value = data[i][1];
        clean_data.push({ axis: axis_name, value: value});
    }
    console.log(clean_data)
        

    // let svg = d3.select("#" + cardEltId).append("svg")
    //     .attr("width", '100%')
    //     .attr("height", '100%')
    //     .attr("id", svgId);

    //Call function to draw the Radar chart
    //RadarChart(
    //      "id or class of to which the new SVG needs to be appended",
    //      the dataset,
    //      the options (optional))
    RadarChart(svgId, [clean_data], radarChartOptions);

//     let rect = document.getElementById(cardEltId).getBoundingClientRect();
//     console.log(rect);
//     let width = rect.right - rect.left;
//     let height = rect.bottom - rect.top;
//     console.log(width, height);


//     // X axis
//     var x = d3.scaleBand()
//         .range([0, width])
//         .domain(data.map(function (d) { return d[0]; }))
//         .padding(0.2);
//     svg.append("g")
//         .attr("transform", "translate(0," + height + ")")
//         .call(d3.axisBottom(x))
//         .selectAll("text")
//         .attr("transform", "translate(-10,0)rotate(-45)")
//         .style("text-anchor", "end");

//     // Add Y axis
//     var y = d3.scaleLinear()
//         .domain([0, 100])
//         .range([height, 0]);
//     svg.append("g")
//         .call(d3.axisLeft(y));

//     // Bars
//     svg.selectAll("mybar")
//         .data(data)
//         .enter()
//         .append("rect")
//         .attr("x", function (d) { return x(d[0]); })
//         .attr("y", function (d) { return y(d[1]); })
//         .attr("width", x.bandwidth())
//         .attr("height", function (d) { return height - y(d[1]); })
//         .attr("fill", "#69b3a2")

}
//  <div class="w-full h-64 bg-gray-600 rounded-lg dark:bg-gray-600 ${eltId}" id="${eltId}"></div>

const CareerCard = (eltId, occupationText, firm, reviewScore, reviewText) => {
    if(firm != "No Match") {
        return ` 

            <div class="w-full drop-shadow-lg">
                <h1 style="text-align: center; margin-left: 50px;" class="w-full h-full centered"><b>${occupationText}</b></h1>

                <div class="${eltId}" id="${eltId}"></div>

                <h2 class="w-full h-full">${firm} (${reviewScore}/5)</h2>
                <p class="w-full h-full">${reviewText}</p>
            </div>`;
    } else {
        return ` 
            <div class="w-full drop-shadow-lg">
                <h1 class="w-full h-full" style="text-align: center; margin-left: 50px;"><b>${occupationText}</b></h1>
                <div class="${eltId}" id="${eltId}"></div>
            </div>`;
    }

};

function CreateCareerCardsTiles(numCards, data) {
    for(let i = 0; i < numCards; i++){
        let cardEltId = 'card' + i;
        let card = document.createElement("div");
        
        cardData = JSON.parse(data[i]);
        console.log(cardEltId, i, cardData);

        card.innerHTML = CareerCard(cardEltId, cardData.job, cardData?.review?.firm, cardData?.review?.average_firm_rating, cardData?.review?.pros);
        cardGrid.appendChild(card);

        drawSpiderChart(cardEltId, cardData.top10);
    }
}


function generateCards(data, cardsToDisplay = 8) {
    if (data.length == 0) {
        return;
    }
    CreateCareerCardsTiles(cardsToDisplay, data);
}

function search() {
    // set loading view

    // get the text
    let text = document.getElementById("search-box").value;
    console.log(text);
    fetch("/search?" + new URLSearchParams({ interest: text }).toString())
        .then((response) => response.json())
        .then((data) => {
            generateCards(data);
        });
}

