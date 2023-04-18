const cardGrid = document.getElementById('card-grid');

function drawSpiderChart(cardEltId, data) {
    let svgId = "svg-" + cardEltId;
    let svg = d3.select("#" + cardEltId).append("svg")
        .attr("width", '100%')
        .attr("height", '100%')
        .attr("id", svgId);

    let rect = document.getElementById(cardEltId).getBoundingClientRect();
    console.log(rect);
    let width = rect.right - rect.left;
    let height = rect.bottom - rect.top;
    console.log(width, height);


    // X axis
    var x = d3.scaleBand()
        .range([0, width])
        .domain(data.map(function (d) { return d[0]; }))
        .padding(0.2);
    svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x))
        .selectAll("text")
        .attr("transform", "translate(-10,0)rotate(-45)")
        .style("text-anchor", "end");

    // Add Y axis
    var y = d3.scaleLinear()
        .domain([0, 100])
        .range([height, 0]);
    svg.append("g")
        .call(d3.axisLeft(y));

    // Bars
    svg.selectAll("mybar")
        .data(data)
        .enter()
        .append("rect")
        .attr("x", function (d) { return x(d[0]); })
        .attr("y", function (d) { return y(d[1]); })
        .attr("width", x.bandwidth())
        .attr("height", function (d) { return height - y(d[1]); })
        .attr("fill", "#69b3a2")

}

const CareerCard = (eltId, occupationText) => {
    return ` 
        <div class="w-full drop-shadow-lg">
            <div class="w-full h-64 bg-gray-300 rounded-lg dark:bg-gray-600" id="${eltId}"></div>

            <h1 class="w-full h-full">${occupationText}</h1>
        </div>`
};

function CreateCareerCardsTiles(numCards, data) {
    for(let i = 0; i < numCards; i++){
        let cardEltId = 'card' + i;
        let card = document.createElement("div");
        
        cardData = JSON.parse(data[i]);
        console.log(cardEltId, i, cardData);

        card.innerHTML = CareerCard(cardEltId, cardData.job);
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



