import * as readline from 'readline';

import { ChatAnthropic } from "@langchain/anthropic";
import { HumanMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";

const model = new ChatAnthropic({
    model: "claude-3-5-sonnet-20240620",
    temperature: 0,
});

const parser = new StringOutputParser();

async function main() {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
    });

    // Create recursive function for the loop
    const askQuestion = async () => {
        rl.question('> ', async (answer) => {
            console.log('--------------------------------');
            if (answer.toLowerCase() === 'x') {
                console.log('Bye!');
                rl.close();
                return;
            }

            const humanMessage = new HumanMessage(answer);
            const messages = [
                humanMessage,
            ];
            const result = await model.invoke(messages);
            const parsedResult = await parser.invoke(result);

            console.log(parsedResult);
            console.log('--------------------------------');
            askQuestion();
        });
    };

    // Start the loop
    await askQuestion();
}

main();


