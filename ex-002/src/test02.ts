import * as readline from 'readline';

import { ChatAnthropic } from "@langchain/anthropic";
import { HumanMessage } from "@langchain/core/messages";
import { ConversationChain } from "langchain/chains";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { BufferMemory } from "langchain/memory";

const model = new ChatAnthropic({
    model: 'claude-3-haiku-20240307',
    temperature: 0,
    verbose: true
});

const memory = new BufferMemory();
const parser = new StringOutputParser();
const chain = new ConversationChain({ llm: model, memory: memory, outputParser: parser });


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

            const response = await chain.call({ input: answer });
            console.log(response.response);
            console.log('--------------------------------');
            askQuestion();
        });
    };

    // Start the loop
    await askQuestion();
}

main();


