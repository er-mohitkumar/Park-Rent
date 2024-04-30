from django_user_agents.utils import get_user_agent

class MobileDesktopMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        user_agent = get_user_agent(request)
        if user_agent.is_mobile:
            request.META['HTTP_USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36'
        
        response = self.get_response(request)
        return response
