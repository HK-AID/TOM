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

        response = orchestrator.process(user_input)
        print(f"TOM: {response}")


if __name__ == "__main__":
    main()
