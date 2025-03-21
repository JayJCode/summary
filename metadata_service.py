from typing import Any
from urllib.parse import parse_qsl
from urllib.parse import urlencode

import requests  # type: ignore
from core.schemas.metadata_models import MetadataMessage
from core.schemas.metadata_models import MetadataResult
from core.services.metadata.metadata_endpoints import MetadataEndpoints
from fastapi import Request
from loguru import logger

from core.utils.summary.summarizer import Summarizer

class MetadataService:
    def __init__(
        self,
        url: str,
        api_version: str,
        api_key_name: str,
        api_key: str,
    ) -> None:
        self._metadata_service_url = url
        self._api_key_name = api_key_name
        self._api_key = api_key
        self._api_version = api_version
        self._base_url = f"{self._metadata_service_url}/{self._api_version}"
        self._extra_query_params_to_remove = ["user_question"]

    def get_metadata(
        self,
        request: Request,
        user_question: str,
        summarize: bool = False,
    ) -> MetadataResult | MetadataMessage:
        """Process metadata request with optional summarization"""
        metadata_response = self._forward_request_to_metadata(request=request)

        if not summarize:
            return metadata_response

        return self._summarize_metadata(metadata_response, user_question)

    def _forward_request_to_metadata(
        self,
        request: Request,
    ) -> MetadataResult | MetadataMessage:
        """Forward an incoming request to the metadata service"""
        method = request.method
        url = self._get_url(request=request)
        headers = self._get_headers()

        logger.info(f"Sending metadata request to {url}")

        response = requests.request(method=method, url=url, headers=headers)
        return self._parse_response(response=response)

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with authentication"""
        return {self._api_key_name: self._api_key}

    def _get_url(self, request: Request) -> str:
        """Build full URL for request forwarding"""
        path_and_query = self._get_path_and_query_url(request)
        separator = "" if path_and_query.startswith("/") else "/"
        return f"{self._metadata_service_url}/{self._api_version}{separator}{path_and_query}"

    def _get_path_and_query_url(
        self,
        request: Request,
        remove_extra_params: bool = True,
    ) -> str:
        """Extract path and query parameters from request"""
        path = request.url.path
        query = request.url.query

        if query and remove_extra_params:
            query = self._remove_extra_query_params(query)

        path_and_query = f"{path}?{query}" if query else path

        logger.info(f"Full URL: {request.url}")
        logger.info(f"Path: {path}")
        logger.info(f"Query: {query}")
        logger.info(f"Path and Query: {path_and_query}")

        return path_and_query

    def _remove_extra_query_params(self, query: str) -> str:
        query_params = parse_qsl(query)
        filtered_params = [
            (name, value)
            for name, value in query_params
            if name not in self._extra_query_params_to_remove
        ]
        return urlencode(filtered_params) if filtered_params else ""

    def _summarize_metadata(self, metadata_response: MetadataResult,
                            user_question: str) -> MetadataResult | MetadataMessage:
        """Summarize metadata based on user question"""
        summarizer = Summarizer()
        # If there are no results, return the metadata response as is
        if not hasattr(metadata_response, 'results') or not metadata_response.results:
            logger.info(f"Data has no results to summarize")
            return metadata_response
        metadata_response_dict = {"results": metadata_response.results}
        return summarizer.summarize(user_question, metadata_response_dict)

    # Enhanced API access methods
    def _build_url(self, endpoint: MetadataEndpoints, **path_params) -> str:
        """
        Build URL for a specific endpoint with path parameters

        Args:
            endpoint: The API endpoint to access
            **path_params: Path parameters to substitute in the URL template

        Returns:
            Complete URL for the API endpoint
        """
        # Format the endpoint path with any provided path parameters
        endpoint_path = (
            endpoint.value.format(**path_params) if path_params else endpoint.value
        )
        return f"{self._base_url}/{endpoint_path}"

    def _request(
        self,
        endpoint: MetadataEndpoints,
        query_params: dict[str, Any] | None = None,
        **path_params,
    ) -> MetadataResult:
        """
        Make a request to a specific endpoint

        Args:
            endpoint: The API endpoint to access
            query_params: Optional query parameters
            **path_params: Path parameters for the endpoint URL

        Returns:
            JSON response data or empty list on error
        """
        url = self._build_url(endpoint, **path_params)

        logger.info(f"Retrieving data from {url}")
        try:
            response = requests.get(
                url,
                headers=self._get_headers(),
                params=query_params,
            )
            response.raise_for_status()
            return self._parse_response(response=response)
        except requests.RequestException as e:
            logger.error(f"Error retrieving data: {e}")
            return self._parse_response(response=response, pass_empty=True)

    def _parse_response(
        self,
        response: requests.Response,
        pass_empty: bool = False,
    ) -> MetadataResult | MetadataMessage:
        """Parse a response object and return JSON data"""
        if pass_empty:
            return MetadataResult(results=[])

        try:
            response_json = response.json()
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {e}")
            return MetadataResult(results=[])

        message_key = "message"
        if any(
            k.lower() == message_key if isinstance(k, str) else k == message_key
            for k in response_json
        ):
            logger.info(f"Metadata error response: {response_json.get(message_key)}")
            return MetadataMessage(message=response_json.get(message_key, ""))

        return MetadataResult(**response_json)

    def _get_results(
        self,
        endpoint: MetadataEndpoints,
        query_params: dict[str, Any] | None = None,
        **path_params,
    ) -> list[Any]:
        """Convenience method that returns just the results array"""
        return self._request(endpoint, query_params, **path_params).results

    def get_tables(self) -> list[dict[str, Any]]:
        """Retrieve all tables"""
        return self._get_results(MetadataEndpoints.TABLES)

    def get_table_metadata(self, table_name: str) -> list[dict[str, Any]]:
        """Retrieve metadata for a specific table"""
        return self._get_results(MetadataEndpoints.TABLE, table_name=table_name)

    def get_table_profiling(self, table_name: str) -> list[dict[str, Any]]:
        """Retrieve profiling data for a specific table"""
        return self._get_results(
            MetadataEndpoints.TABLE_PROFILING,
            table_name=table_name,
        )

    def get_table_attribute(
        self,
        table_name: str,
        attribute_name: str,
    ) -> list[dict[str, Any]]:
        """Retrieve attribute metadata for a specific table attribute"""
        return self._get_results(
            MetadataEndpoints.TABLE_ATTRIBUTE,
            table_name=table_name,
            attribute_name=attribute_name,
        )

    def get_table_attribute_profiling(
        self,
        table_name: str,
        attribute_name: str,
    ) -> list[dict[str, Any]]:
        """Retrieve profiling data for a specific table attribute"""
        return self._get_results(
            MetadataEndpoints.TABLE_ATTRIBUTE_PROFILING,
            table_name=table_name,
            attribute_name=attribute_name,
        )

    def get_table_attributes(self, table_name: str) -> list[dict[str, Any]]:
        """Retrieve all attributes for a specific table"""
        return self._get_results(
            MetadataEndpoints.TABLE_ATTRIBUTES,
            table_name=table_name,
        )

    def get_counts(self, **query_params) -> dict[str, Any]:
        """Retrieve counts based on query parameters"""
        return self._request(MetadataEndpoints.COUNTS, query_params=query_params)

    def search_metadata(
        self,
        search_term: str,
        **additional_params,
    ) -> list[dict[str, Any]]:
        """Search metadata using a search term"""
        params = {"q": search_term, "index": "metadata", **additional_params}
        return self._get_results(MetadataEndpoints.SEARCH, query_params=params)

    def get_aggregate(self, **query_params) -> dict[str, Any]:
        """Retrieve aggregated item count based on query parameters"""
        return self._request(MetadataEndpoints.AGGREGATE, query_params=query_params)

    def get_schemas(self) -> list[dict[str, Any]]:
        """Retrieve all schemas"""
        return self._get_results(MetadataEndpoints.SCHEMAS)

    def get_schema_tables(self, schema_name: str) -> list[dict[str, Any]]:
        """Retrieve all tables for a specific schema"""
        return self._get_results(
            MetadataEndpoints.SCHEMA_TABLES,
            schema_name=schema_name,
        )

    def get_schema_table_metadata(
        self,
        schema_name: str,
        table_name: str,
    ) -> list[dict[str, Any]]:
        """Retrieve metadata for a specific table in a specific schema"""
        return self._get_results(
            MetadataEndpoints.SCHEMA_TABLE,
            schema_name=schema_name,
            table_name=table_name,
        )

    def health_check(self) -> dict[str, Any]:
        """Check the health status of the metadata service"""
        return self._request(MetadataEndpoints.HEALTH)
