import os
import asyncio
import redis.asyncio as redis
import signal
from typing import List
from pydantic import BaseModel
from fastembed import TextEmbedding


class EmbeddingRequest(BaseModel):
    documents: List[str]


REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_DB = int(os.environ.get("REDIS_DB", "0"))

canceled = False
embedding_model = TextEmbedding()
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


async def main():
    await asyncio.gather(health_check(), process_embeddings())
    await redis_client.aclose()


async def process_embeddings():
    async with redis_client.pubsub() as pubsub:
        await pubsub.subscribe("ai.embeddings")

        while not canceled:
            msg = await pubsub.get_message(ignore_subscribe_messages=True)
            if msg is not None:
                try:
                    req = EmbeddingRequest.model_validate_json(msg["data"].decode())
                    embeddings = list(embedding_model.embed(req.documents))
                    print(
                        f"Embedding: len={len(embeddings)}, head={embeddings[0][:5]}..."
                    )
                except Exception as e:
                    print(f"Error: {e}")


async def health_check():
    while not canceled:
        await asyncio.sleep(1)
        await redis_client.publish("ai.heartbeat", "hello")


if __name__ == "__main__":
    print("Hello, World!")
    try:
        loop = asyncio.get_event_loop()
        task = asyncio.ensure_future(main())
        shielded_task = asyncio.shield(task)

        def shutdown():
            print("Shutting down...")
            global canceled
            canceled = True

        loop.add_signal_handler(signal.SIGINT, shutdown)
        loop.add_signal_handler(signal.SIGTERM, shutdown)

        loop.run_until_complete(shielded_task)
    except asyncio.CancelledError:
        pass
