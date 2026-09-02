"""Main entrypoint for TOM terminal interface."""
import sys
from core.orchestrator import TOMOrchestrator


def main() -> None:
    orchestrator = TOMOrchestrator()
    print("TOM: Online")

    while True:
        try:
            user_input = input("User: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nTOM: Shutting down.")
            sys.exit(0)

        if user_input.lower() == "exit":
            print("TOM: Goodbye.")
            break

        if not user_input:
            continue

        print("TOM: ", end="", flush=True)
        for chunk in orchestrator.stream(user_input):
            print(chunk, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
