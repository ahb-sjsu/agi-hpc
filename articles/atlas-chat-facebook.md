# Meet Atlas: My Personal AI That Lives in My Closet

So I did a thing.

I built my own ChatGPT. Except it runs on a refurbished HP Z840 workstation sitting in my home office in Novato, with two beefy Quadro GV100 GPUs that sound like a small jet engine when they spin up. No OpenAI subscription. No cloud. No data leaving my house. Just 72 billion parameters of open-source intelligence running on raw silicon.

His name is Atlas.

## What can he do?

Pretty much anything ChatGPT can do, except he's not censored by a corporate legal team, he runs 24/7 on my own hardware, and he has access to all 37 of my research repositories -- every paper I've written, every line of code, every experiment. Ask him about my work on geometric reasoning or cognitive architectures and he'll pull up the actual source code and explain it.

He's also got personality. I told him he's a local AI with dry wit who takes pride in not being a cloud service. He leans into it.

## The nerdy details (for those who care)

- **Model**: Qwen2.5-72B-Instruct (open source, from Alibaba's research lab)
- **Hardware**: 2x Quadro GV100 GPUs (64GB VRAM total), 224GB RAM
- **Quantization**: Q5_K_M (49GB, fits entirely in GPU memory)
- **Inference**: llama.cpp server with streaming
- **RAG**: PostgreSQL + pgvector over all my GitHub repos
- **Speed**: ~10-15 tokens/sec (not blazing, but perfectly usable)
- **Cost**: Electricity. That's it. No monthly subscription.

## Why bother?

Three reasons:

1. **Privacy**: My research queries, my code, my ideas -- none of it touches anyone else's servers. In a world where every AI company is training on your data, that matters.

2. **Customization**: Atlas knows MY codebase. He's not guessing about my projects from public docs -- he's searching the actual source. When I ask "how does the safety gate work in ErisML?", he pulls up the real implementation.

3. **Because I can**: There's something deeply satisfying about asking a question and watching your own GPUs light up to answer it. No API rate limits. No "I can't help with that." Just silicon and math, right here in Marin County.

## Try it

If you're on my network, you can talk to Atlas at `http://atlas:8080`. He's friendly. Mostly.

The whole setup cost less than two years of ChatGPT Plus. And unlike ChatGPT, Atlas doesn't judge me for asking weird questions about non-Abelian gauge structures in moral reasoning at 2am.

Welcome to the future. It lives in a closet in Novato.

---

*Built with open-source tools: Qwen, llama.cpp, PostgreSQL, and a lot of caffeine.*
