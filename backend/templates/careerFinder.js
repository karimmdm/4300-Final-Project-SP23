const CareerCard = () => {
    var skillsDiagram = React.createElement("div", { "class": "w-full h-64 bg-gray-300 rounded-lg dark:bg-gray-600" });
    var careerTitle = React.createElement("h1", { "class": "w-56 h-2 mt-4 bg-gray-200 rounded-lg dark:bg-gray-700" });
    var card = React.createElement("div", { "class": "w-full" }, [skillsDiagram, careerTitle]);
    return card;
};

const CareerCardGrid = () => {
    var cards = [];
    for (let i = 0; i < 8; i++) {
        cards.push(CareerCard());
    }
    return (
        <div class="grid grid-cols-1 gap-8 mt-8 xl:mt-12 xl:gap-12 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 animate-pulse">{cards}</div>
    );
}

ReactDOM.render(React.createElement(CareerCardGrid), document.getElementById("careers-container"));