const { spawn } = require("child_process");

/**
 * Example of Python tenserflow project implemented into js
 */

class textClassifier {
    pyprocess = null;
    isReady = false;

    clasify = (message) => {
        if (!this.isReady) throw new Error("The clasifier is not ready");
        this.pyprocess.stdin.write(message + "\n");
        return new Promise((res) => {
            this.pyprocess.stdout.on("data", (message) => {
                const data = this.messageToJSON(message);
                if (!data) return;
                if (data.event == "job_complete") res(data.data);
            });
        });
    };

    messageToJSON = (d) => {
        const message = d.toString().trim();
        if (!message.startsWith("MESSAGE::")) return;
        let data = undefined;

        try {
            data = JSON.parse(message.substring("MESSAGE::".length));
        } catch (_) {}

        if (!data) return;
        return data;
    };

    start = async () => {
        if (this.pyprocess) {
            if (!this.pyprocess.killed) this.pyprocess.kill();
            this.pyprocess = null;
        }

        this.pyprocess = spawn("python", [
            "main.py",
            "-m",
            "./models/0",
            "-std",
            "true",
        ]);

        await new Promise((res, rej) => {
            setTimeout(() => {
                this.pyprocess = null;
                this.isReady = false;
                rej("Couldnt load the model");
            }, 10000);
            let loaded_prop = false;
            let loaded_model = false;
            this.pyprocess.stdout.on("data", (message) => {
                const data = this.messageToJSON(message);
                if (!data) return;
                if (data.event == "load_model") loaded_model = true;
                if (data.event == "load_model_proprieties") loaded_prop = true;

                if (loaded_model && loaded_prop) {
                    this.isReady = true;
                    res();
                }
            });
        });
    };
}

const clasifier = new textClassifier();
clasifier
    .start()
    .then(async () => {
        const data = await clasifier.clasify("I forgot my password...");
        console.log(data);
        /*
            Example output:
            {
                query: 'i forgot my password',
                label: 'forgot_cedentials',
                accuracy: '0.63736045'
            }
        */
    })
    .catch(console.log);
