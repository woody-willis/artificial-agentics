import "dotenv/config";

import {
    DevelopmentTeamManager,
    DevelopmentTeamManagerInvocationTask,
} from "./teams/development/team-manager";

(async () => {
    const teamManager = await new DevelopmentTeamManager().init();
    console.log("Team manager initialized");

    console.log("Invoking add feature task");
    await teamManager
        .invoke({
            task: DevelopmentTeamManagerInvocationTask.AddFeature,
            data: {
                description:
                    "Add a new agent to do code reviews on a specific branch. The agent should be able to review code changes, suggest improvements, and ensure coding standards are met. Add the agent in src/teams/development/code-reviewer.ts and ensure that the agent structure is the same as other agents in the development team.",
            },
        })
        .catch((error) => {
            console.error("Error invoking add feature task:", error);
        });
    console.log("Add feature task complete");

    await teamManager.dispose();
})();
