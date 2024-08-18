import os
import asyncio
import nats
import signal
from typing import List
from pydantic import BaseModel
from fastembed import TextEmbedding


class EmbeddingRequest(BaseModel):
    documents: List[str]


NATS_SERVERS = os.environ.get("NATS_URL", "nats://localhost:4222").split(",")

canceled = False
embedding_model = TextEmbedding()


async def main():
    nc = await nats.connect(servers=NATS_SERVERS)

    await asyncio.gather(health_check(nc), process_embeddings(nc))

    print("Draining...")
    await nc.drain()


async def process_embeddings(nc):
    sub = await nc.subscribe("ai.embeddings")

    while not canceled:
        try:
            msg = await sub.next_msg(timeout=0.1)
            try:
                req = EmbeddingRequest.model_validate_json(msg.data.decode())
                embeddings = list(embedding_model.embed(req.documents))
                print(f"Embedding: len={len(embeddings)}, head={embeddings[0][:5]}...")
            except Exception as e:
                print(f"Error: {e}")

        except nats.errors.TimeoutError:
            pass

    await sub.unsubscribe()


async def health_check(nc):
    while not canceled:
        await asyncio.sleep(1)
        await nc.publish("ai.heartbeat", b"hello")


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
