from app import create_app
import pytest


def test_case():
    app = create_app()
    app.test_request_context()


if __name__ == '__main__':
    test_case()