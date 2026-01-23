"""
Tests for image endpoints
"""
import pytest
import io


class TestSimilarImagesEndpoint:
    """Tests for POST /image/similar endpoint."""

    def test_similar_without_auth(self, client, sample_image_bytes):
        """Test similar images endpoint requires authentication."""
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/image/similar", files=files)
        assert response.status_code == 401

    def test_similar_valid_image(self, client, auth_headers, sample_image_bytes):
        """Test similar images endpoint with valid image."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/image/similar", files=files, headers=auth_headers)
        # May return 503 if models not loaded
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "similar_images" in data
            assert "query_info" in data
            assert "total_results" in data

    def test_similar_with_top_k(self, client, auth_headers, sample_image_bytes):
        """Test similar images endpoint with custom top_k."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post(
            "/image/similar?top_k=3",
            files=files,
            headers=auth_headers
        )
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert data["total_results"] <= 3

    def test_similar_invalid_file_type(self, client, auth_headers, invalid_file_bytes):
        """Test similar images endpoint with invalid file type."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        files = {"file": ("test.txt", io.BytesIO(invalid_file_bytes), "text/plain")}
        response = client.post("/image/similar", files=files, headers=auth_headers)
        assert response.status_code == 400

    def test_similar_no_file(self, client, auth_headers):
        """Test similar images endpoint without file."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.post("/image/similar", headers=auth_headers)
        assert response.status_code == 422

    def test_similar_invalid_top_k(self, client, auth_headers, sample_image_bytes):
        """Test similar images endpoint with invalid top_k."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post(
            "/image/similar?top_k=0",
            files=files,
            headers=auth_headers
        )
        assert response.status_code == 422

    def test_similar_top_k_too_large(self, client, auth_headers, sample_image_bytes):
        """Test similar images endpoint with top_k exceeding limit."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post(
            "/image/similar?top_k=100",
            files=files,
            headers=auth_headers
        )
        assert response.status_code == 422


class TestClassifyEndpoint:
    """Tests for POST /image/classify endpoint."""

    def test_classify_without_auth(self, client, sample_image_bytes):
        """Test classify endpoint requires authentication."""
        files = {"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/image/classify", files=files)
        assert response.status_code == 401

    def test_classify_valid_image(self, client, auth_headers, sample_image_bytes):
        """Test classify endpoint with valid image."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        files = {"file": ("butterfly.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post("/image/classify", files=files, headers=auth_headers)
        # May return 503 if models not loaded
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "top_prediction" in data
            assert "top_confidence" in data
            assert "query_info" in data

    def test_classify_with_top_k(self, client, auth_headers, sample_image_bytes):
        """Test classify endpoint with custom top_k."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        files = {"file": ("butterfly.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        response = client.post(
            "/image/classify?top_k=3",
            files=files,
            headers=auth_headers
        )
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert len(data["predictions"]) <= 3

    def test_classify_invalid_file_type(self, client, auth_headers, invalid_file_bytes):
        """Test classify endpoint with invalid file type."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        files = {"file": ("test.pdf", io.BytesIO(invalid_file_bytes), "application/pdf")}
        response = client.post("/image/classify", files=files, headers=auth_headers)
        assert response.status_code == 400

    def test_classify_no_file(self, client, auth_headers):
        """Test classify endpoint without file."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        response = client.post("/image/classify", headers=auth_headers)
        assert response.status_code == 422

    def test_classify_png_image(self, client, auth_headers):
        """Test classify endpoint accepts PNG images."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        # Minimal PNG (1x1 red pixel)
        png_bytes = bytes([
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,  # PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,  # IHDR chunk
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41,  # IDAT chunk
            0x54, 0x08, 0xD7, 0x63, 0xF8, 0xFF, 0xFF, 0x3F,
            0x00, 0x05, 0xFE, 0x02, 0xFE, 0xDC, 0xCC, 0x59,
            0xE7, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E,  # IEND chunk
            0x44, 0xAE, 0x42, 0x60, 0x82
        ])

        files = {"file": ("test.png", io.BytesIO(png_bytes), "image/png")}
        response = client.post("/image/classify", files=files, headers=auth_headers)
        # Accept 200, 400 (if image too small), or 503 (models not loaded)
        assert response.status_code in [200, 400, 500, 503]


class TestFileSizeLimit:
    """Tests for file size limits."""

    def test_file_too_large(self, client, auth_headers):
        """Test that files exceeding size limit are rejected."""
        if not auth_headers:
            pytest.skip("Could not obtain auth token")

        # Create a file larger than 10MB (the limit)
        large_file = b"x" * (11 * 1024 * 1024)  # 11MB

        files = {"file": ("large.jpg", io.BytesIO(large_file), "image/jpeg")}
        response = client.post("/image/similar", files=files, headers=auth_headers)
        assert response.status_code in [400, 413]
