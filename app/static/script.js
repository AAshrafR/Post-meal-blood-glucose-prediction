const form = document.getElementById("prediction-form");

const resultBox = document.getElementById("result");
const errorBox = document.getElementById("error");

const predictionValue = document.getElementById(
    "prediction-value"
);

const predictButton = document.getElementById(
    "predict-button"
);


form.addEventListener("submit", async function (event) {

    event.preventDefault();

    resultBox.classList.add("hidden");
    errorBox.classList.add("hidden");

    predictButton.disabled = true;
    predictButton.textContent = "Predicting...";


    const mealDate = document.getElementById(
        "meal_date"
    ).value;

    const mealTime = document.getElementById(
        "meal_time"
    ).value;


    const data = {

        age: Number(
            document.getElementById("age").value
        ),

        gender: document.getElementById(
            "gender"
        ).value,

        bmi: Number(
            document.getElementById("bmi").value
        ),

        weight_lb: Number(
            document.getElementById("weight_lb").value
        ),

        height_in: Number(
            document.getElementById("height_in").value
        ),

        a1c: Number(
            document.getElementById("a1c").value
        ),

        fasting_glucose_lab: Number(
            document.getElementById(
                "fasting_glucose_lab"
            ).value
        ),

        meal_hour: Number(
            document.getElementById("meal_hour").value
        ),

        meal_type: document.getElementById(
            "meal_type"
        ).value,

        calories: Number(
            document.getElementById("calories").value
        ),

        carbs: Number(
            document.getElementById("carbs").value
        ),

        protein: Number(
            document.getElementById("protein").value
        ),

        fat: Number(
            document.getElementById("fat").value
        ),

        fiber: Number(
            document.getElementById("fiber").value
        ),

        amount_consumed: Number(
            document.getElementById(
                "amount_consumed"
            ).value
        ),

        premeal_glucose: Number(
            document.getElementById(
                "premeal_glucose"
            ).value
        ),

        glucose_sensor_used: document.getElementById(
            "glucose_sensor_used"
        ).value,

        meal_time: `${mealDate} ${mealTime}:00`
    };


    try {

        const response = await fetch(
            "/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(data)
            }
        );


        const result = await response.json();


        if (!response.ok) {
            throw new Error(
                result.error || "Prediction failed."
            );
        }


        predictionValue.textContent =
            Number(
                result.predicted_glucose_120min
            ).toFixed(2);


        resultBox.classList.remove("hidden");


    } catch (error) {

        errorBox.textContent =
            error.message;

        errorBox.classList.remove("hidden");

    } finally {

        predictButton.disabled = false;
        predictButton.textContent =
            "Predict Glucose";
    }

});